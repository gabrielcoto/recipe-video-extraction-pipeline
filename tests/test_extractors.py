import json
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path

import httpx
import pytest

from src import downloader, media
from src.downloader import SSRFBlockedError, UnsupportedPlatformError
from src.extractors import frames as frames_mod
from src.extractors import ocr as ocr_mod
from src.extractors import stt as stt_mod
from src.llm import fusion
from src.errors import MediaProcessingError
from src.config import WHISPER_LANGUAGE
from src.models import UploadSource


def test_downloader_rejects_unsupported_platform():
    with pytest.raises(UnsupportedPlatformError) as excinfo:
        downloader.probe_duration("https://example.org/videos/pasta")
    assert "unsupported_platform" == excinfo.value.code


def test_downloader_blocks_ssrf(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SSRFBlockedError):
        downloader.probe_duration("https://youtu.be/abc123")


def test_downloader_rejects_non_http_scheme():
    with pytest.raises(UnsupportedPlatformError):
        downloader.probe_duration("ftp://youtube.com/video")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_extract_frames_limits_frame_count(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", str(video)],
        check=True, capture_output=True, text=True,
    )
    frames = frames_mod.extract_frames(video, tmp_path / "frames")
    assert 1 <= len(frames) <= 12


class _FakeReader:
    """Эмулирует PaddleOCR.ocr: один вызов = один ответ для одного изображения.

    Реальный PaddleOCR возвращает список страниц: ``[page]`` для одного кадра,
    где ``page`` — список регионов ``[box, (text, conf)]``. Тест передаёт
    список ответов по одному на каждый кадр и раздаёт их по порядку вызовов.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def ocr(self, path, cls=True):  # noqa: ARG002 - интерфейс PaddleOCR
        if self._idx >= len(self._responses):
            raise AssertionError(f"_FakeReader: слишком много вызовов ocr() для path={path!r}")
        response = self._responses[self._idx]
        self._idx += 1
        return response


def test_ocr_deduplicates_lines_in_order(monkeypatch):
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    pages = [
        [[[box, ("Яйца 2 шт", 0.9)]]],
        [[[box, ("яйца 2 шт", 0.8)], [box, ("Соль", 0.7)]]],
    ]
    monkeypatch.setattr(ocr_mod, "_get_reader", lambda: _FakeReader(pages))
    text = ocr_mod.recognize_text([Path("a.jpg"), Path("b.jpg")])
    assert text.splitlines() == ["Яйца 2 шт", "Соль"]


def test_ocr_confidence_threshold_drops_weak_lines(monkeypatch):
    # OCR_MIN_CONFIDENCE отбрасывает мусорные прочтения ДО дедупликации:
    # слабая строка не попадает ни в выдачу, ни в seen, поэтому её же
    # уверенное повторение на другом кадре сохраняется. Порог 0 (по
    # умолчанию) поведение не меняет — см. тест выше.
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    pages = [
        [[[box, ("Salt", 0.3)], [box, ("Flour", 0.9)]]],
        [[[box, ("salt", 0.85)]]],
    ]
    monkeypatch.setattr(ocr_mod, "OCR_MIN_CONFIDENCE", 0.6)
    monkeypatch.setattr(ocr_mod, "_get_reader", lambda: _FakeReader(pages))
    text = ocr_mod.recognize_text([Path("a.jpg"), Path("b.jpg")])
    assert text.splitlines() == ["Flour", "salt"]


def test_ocr_wrapper_disables_mkldnn_on_every_predictor():
    # Регрессия: Paddle 3.x включает oneDNN для CPU по умолчанию
    # (Config().mkldnn_enabled() == True), а PaddleOCR 2.x при enable_mkldnn=False
    # не вызывает disable_mkldnn() — на части CPU и в эмуляции это даёт
    # «could not create a primitive descriptor for a reorder primitive».
    # Обёртка create_predictor обязана выключать oneDNN для каждого предсказателя.
    calls: list[str] = []

    class FakeConfig:
        def disable_mkldnn(self):
            calls.append("disable")

    def fake_create_predictor(config, *args, **kwargs):  # noqa: ARG001 - форма вызова paddle
        calls.append("create")
        return "predictor"

    module = types.SimpleNamespace(create_predictor=fake_create_predictor)
    ocr_mod._wrap_create_predictor_without_onednn(module)
    assert module.create_predictor(FakeConfig()) == "predictor"
    assert calls == ["disable", "create"]  # сначала выключили oneDNN, потом создали предсказатель

    # Повторный вызов не оборачивает предсказатель второй раз (нет двойного disable).
    wrapper = module.create_predictor
    ocr_mod._wrap_create_predictor_without_onednn(module)
    assert module.create_predictor is wrapper


def _valid_payload():
    return {
        "schema_version": "1.0.0",
        "title": "Омлет",
        "servings": 1,
        "ingredients": [{"name": "Яйца", "quantity": {"value": 2, "unit": "pcs"}, "note": None}],
        "steps": [{"instruction": "Взбейте яйца.", "duration_seconds": None}],
        "total_time_minutes": None,
    }


def test_fusion_returns_valid_recipe_and_keeps_server_source(monkeypatch):
    monkeypatch.setattr(
        fusion, "_call_llm", lambda messages: "```json\n" + json.dumps(_valid_payload()) + "\n```"
    )
    recipe = fusion.fuse_recipe("взбейте два яйца", "яйца 2 шт", UploadSource(type="upload"))
    assert recipe.title == "Омлет"
    assert recipe.source.type == "upload"


def test_fusion_retries_with_schema_feedback(monkeypatch):
    attempts = []

    def fake_llm(messages):
        attempts.append(list(messages))
        if len(attempts) == 1:
            return "Извините, вот рецепт без JSON"
        bad = _valid_payload()
        bad.pop("steps")
        return json.dumps(bad)

    monkeypatch.setattr(fusion, "_call_llm", fake_llm)
    with pytest.raises(fusion.LLMInvalidResponseError):
        fusion.fuse_recipe("текст", "текст", UploadSource(type="upload"))
    assert len(attempts) == 3  # LLM_MAX_RETRIES=2 -> всего 3 попытки
    # Каждая следующая попытка получает конкретную обратную связь о схеме.
    assert "не прошёл проверку" in attempts[1][-1]["content"]
    assert "steps" in str(attempts[2][-1]["content"])


def test_fusion_rejects_empty_inputs():
    with pytest.raises(fusion.NoUsableContentError):
        fusion.fuse_recipe("   ", "", UploadSource(type="upload"))


def test_fusion_prompt_language_follows_video_by_default(monkeypatch):
    # Регрессия: раньше промпт жёстко требовал русский ответ («Язык ответа — русский»),
    # и никакая переменная окружения на это не влияла. По умолчанию (auto) язык
    # рецепта определяется языком исходного видео.
    monkeypatch.setattr(fusion, "OUTPUT_LANGUAGE", "auto")
    received: dict = {}

    def fake_llm(messages):
        received["system"] = messages[0]["content"]
        return json.dumps(_valid_payload())

    monkeypatch.setattr(fusion, "_call_llm", fake_llm)
    fusion.fuse_recipe("текст", "текст", UploadSource(type="upload"))
    assert "язык исходного видео" in received["system"]
    assert "Язык ответа — русский" not in received["system"]


def test_fusion_prompt_can_force_output_language(monkeypatch):
    # OUTPUT_LANGUAGE=English задаёт язык рецепта независимо от языка входных данных.
    monkeypatch.setattr(fusion, "OUTPUT_LANGUAGE", "English")
    prompt = fusion._system_prompt()
    assert "Язык ответа — English" in prompt
    assert "даже если входные данные на другом языке" in prompt


def test_fusion_prompt_keeps_units_within_contract_1_0_0():
    # Промпт ограничен восемью единицами контракта 1.0.0: расширение Unit
    # (cup, oz, lb) — черновик 1.1.0, ожидающий согласования upstream
    # (docs/drafts/). До одобрения cup/oz/lb пересчитываются промптом по
    # фиксированным коэффициентам с сохранением исходной формулировки в note,
    # «стакан» и прочие меры без коэффициента — только note. Английские
    # счётные слова (clove, slice, can, stick) — те же уточнения: pcs + note,
    # видео на согласованных платформах англоязычные.
    units_line = next(line for line in fusion._system_prompt().splitlines()
                      if "Единицы только из списка" in line)
    for unit in ("cup", "fl oz", "oz", "lb", "clove", "slice", "stick", "can"):
        assert unit not in units_line
    prompt = fusion._system_prompt()
    assert "g, kg, ml, l, pcs, tsp, tbsp, pinch" in units_line
    assert '«зубчик», «ломтик», «пачка», «банка»' in prompt
    assert "clove, slice, can, stick" in prompt
    assert '"1 stick of butter"' in prompt
    assert "1 cup = 240 ml" in prompt
    assert "1 oz = 28.35 g" in prompt
    assert "1 lb = 453.6 g" in prompt
    assert '"1 cup flour"' in prompt
    assert '"1 стакан муки"' in prompt


def test_fusion_prompt_requires_minor_ingredients_and_note_fallback():
    # Регрессия: на плотной выдаче OCR (MAX_FRAMES=24) модель пропускала мелкие
    # позиции вроде соли; единицы вне списка теряли количество целиком.
    prompt = fusion._system_prompt()
    assert "соль" in prompt
    assert "каждый ингредиент" in prompt
    assert "quantity: null" in prompt
    assert "в note" in prompt


def test_fusion_sends_configured_temperature(monkeypatch):
    # Регрессия: температура была жёстко 0.0 в теле запроса — настроить её было нельзя.
    received: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(_valid_payload())}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ARG002 - форма вызова httpx
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):  # noqa: A002 - параметр httpx
            received["url"] = url
            received["json"] = json
            return FakeResponse()

    monkeypatch.setattr(fusion.httpx, "Client", FakeClient)
    monkeypatch.setattr(fusion, "LLM_API_KEY", "key")
    monkeypatch.setattr(fusion, "LLM_BASE_URL", "https://llm.example")
    monkeypatch.setattr(fusion, "LLM_MODEL", "test-model")
    monkeypatch.setattr(fusion, "LLM_TEMPERATURE", 0.3)

    recipe = fusion.fuse_recipe("взбейте два яйца", "яйца 2 шт", UploadSource(type="upload"))

    assert recipe.title == "Омлет"
    assert received["json"]["temperature"] == 0.3
    assert received["json"]["model"] == "test-model"
    assert received["url"] == "https://llm.example/chat/completions"


def _fake_llm_client(monkeypatch, behaviors):
    """FakeClient, чей post() раздаёт поведения по одному на каждый вызов.

    Поведение — либо экземпляр httpx.TransportError (сетевой сбой), либо
    HTTP-статус (int); ответ 200 несёт валидный Recipe. Последнее поведение
    повторяется, если вызовов больше, чем элементов списка. Возвращает счётчик
    вызовов post(). Пауза между повторами обнуляется (fusion.sleep).
    """
    state = {"calls": 0}

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(_valid_payload())}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ARG002 - форма вызова httpx
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):  # noqa: A002 - параметр httpx
            behavior = behaviors[min(state["calls"], len(behaviors) - 1)]
            state["calls"] += 1
            if isinstance(behavior, Exception):
                raise behavior
            return FakeResponse(behavior)

    monkeypatch.setattr(fusion.httpx, "Client", FakeClient)
    monkeypatch.setattr(fusion, "LLM_API_KEY", "key")
    monkeypatch.setattr(fusion, "LLM_BASE_URL", "https://llm.example")
    monkeypatch.setattr(fusion, "LLM_MODEL", "test-model")
    monkeypatch.setattr(fusion, "sleep", lambda seconds: None)
    return state


def test_fusion_retries_transient_network_error(monkeypatch):
    # Регрессия: сетевой сбой (обрыв соединения, таймаут) раньше вызывал
    # LLMUnavailableError сразу, и один сбой провайдера срывал всю загрузку.
    # Транзиторные сетевые сбои повторяются отдельно от повторов схемы
    # (LLM_MAX_RETRIES покрывает только невалидный JSON/нарушение схемы).
    calls = _fake_llm_client(monkeypatch, [httpx.ConnectError("connection reset"), 200])
    monkeypatch.setattr(fusion, "LLM_NETWORK_RETRIES", 1)
    recipe = fusion.fuse_recipe("взбейте два яйца", "яйца 2 шт", UploadSource(type="upload"))
    assert recipe.title == "Омлет"
    assert calls["calls"] == 2


@pytest.mark.parametrize("status", [408, 429, 502, 503, 504])
def test_fusion_retries_transient_http_statuses(monkeypatch, status):
    # Транзиторные статусы шлюза повторяются: 502 от OpenAI-совместимого
    # эндпоинта часто временный и не должен ронять всю загрузку.
    calls = _fake_llm_client(monkeypatch, [status, 200])
    monkeypatch.setattr(fusion, "LLM_NETWORK_RETRIES", 1)
    recipe = fusion.fuse_recipe("текст", "текст", UploadSource(type="upload"))
    assert recipe.title == "Омлет"
    assert calls["calls"] == 2


def test_fusion_does_not_retry_permanent_http_status(monkeypatch):
    # 400/401/403/404 от повтора не меняются — падать нужно сразу, чтобы не
    # добавлять к фейловому запросу паузы и лишние вызовы.
    calls = _fake_llm_client(monkeypatch, [401])
    with pytest.raises(fusion.LLMUnavailableError, match="HTTP 401"):
        fusion.fuse_recipe("текст", "текст", UploadSource(type="upload"))
    assert calls["calls"] == 1


def test_fusion_network_retry_exhaustion_is_bounded(monkeypatch):
    # Число сетевых повторов ограничено (требование рубрики: «число повторов
    # LLM ограничено»): после исчерпания — LLMUnavailableError, а не вечные попытки.
    calls = _fake_llm_client(monkeypatch, [httpx.ReadTimeout("timed out")])
    monkeypatch.setattr(fusion, "LLM_NETWORK_RETRIES", 2)
    with pytest.raises(fusion.LLMUnavailableError, match="недоступен"):
        fusion.fuse_recipe("текст", "текст", UploadSource(type="upload"))
    assert calls["calls"] == 3  # 1 исходная попытка + 2 повтора


def _render_video(tmp_path: Path, name: str, with_audio: bool) -> Path:
    """Реальный MP4 через lavfi: с звуковой дорожкой (sine) или без неё (-an)."""
    command = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    command += ["-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac", "-shortest"] if with_audio else ["-an"]
    path = tmp_path / name
    command += [str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_run_media_raises_on_ffmpeg_failure(tmp_path):
    # Регрессия: run_media использовал check=False и пропускал неуспешный
    # код возврата, из-за чего вызывающий код падал на несозданных файлах.
    with pytest.raises(RuntimeError, match="ffmpeg"):
        media.run_media(["ffmpeg", "-y", "-i", str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4")])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_transcribe_returns_empty_for_video_without_audio(monkeypatch, tmp_path):
    # Регрессия: у видео без аудиодорожки ffmpeg-извлечение звука падает,
    # старый код шёл дальше и читал несозданный wav. Теперь — пустая расшифровка,
    # и рецепт собирается по OCR; STT-модель не нужна.
    video = _render_video(tmp_path, "muted.mp4", with_audio=False)
    monkeypatch.setattr(
        stt_mod,
        "_get_model",
        lambda: pytest.fail("STT-модель не должна создаваться для видео без аудиодорожки"),
    )
    assert stt_mod.transcribe(video) == ""


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_transcribe_hands_extracted_wav_to_model(monkeypatch, tmp_path):
    # Видео со звуком доходит до модели с настоящим wav от ffmpeg.
    # Проверять файл нужно в момент вызова: временная папка удаляется на выходе.
    video = _render_video(tmp_path, "voiced.mp4", with_audio=True)
    received: dict = {}

    class FakeModel:
        def transcribe(self, path, language, vad_filter):
            wav = Path(path)
            received["size"] = wav.stat().st_size  # FileNotFoundError, если wav не создан
            received["language"] = language
            return iter([]), None  # нет сегментов -> пустая расшифровка

    monkeypatch.setattr(stt_mod, "_get_model", lambda: FakeModel())
    assert stt_mod.transcribe(video) == ""
    assert received["size"] > 44  # реальный wav от ffmpeg, а не отсутствующий путь
    assert received["language"] == WHISPER_LANGUAGE


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_transcribe_auto_language_passes_none_to_whisper(monkeypatch, tmp_path):
    # WHISPER_LANGUAGE=auto — автодетект языка речи faster-whisper: в
    # transcribe уходит language=None. Раньше строка конфига уходила как
    # есть, и "auto" ловил ошибку валидации языка вместо автодетекта.
    video = _render_video(tmp_path, "voiced-auto.mp4", with_audio=True)
    received: dict = {}

    class FakeModel:
        def transcribe(self, path, language, vad_filter):  # noqa: ARG002 - форма вызова
            received["language"] = language
            return iter([]), None

    monkeypatch.setattr(stt_mod, "WHISPER_LANGUAGE", "auto")
    monkeypatch.setattr(stt_mod, "_get_model", lambda: FakeModel())
    assert stt_mod.transcribe(video) == ""
    assert received["language"] is None


def test_transcribe_reports_unreadable_file(tmp_path):
    # Нечитаемый файл — MediaProcessingError по контракту ошибок, а не сырый RuntimeError.
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"1234")
    with pytest.raises(MediaProcessingError):
        stt_mod.transcribe(bad)


def test_stt_uses_configured_model_settings(monkeypatch):
    # Регрессия: compute_type был жёстко "int8" — задать float16 на GPU или
    # float32 было невозможно; теперь все три параметра модели читаются из конфига.
    calls: dict = {}

    class FakeWhisperModel:
        def __init__(self, model, device=None, compute_type=None):
            calls.update(model=model, device=device, compute_type=compute_type)

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(stt_mod, "_MODEL", None)
    monkeypatch.setattr(stt_mod, "WHISPER_MODEL", "medium")
    monkeypatch.setattr(stt_mod, "WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(stt_mod, "WHISPER_COMPUTE_TYPE", "float16")

    stt_mod._get_model()

    assert calls == {"model": "medium", "device": "cuda", "compute_type": "float16"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
def test_extract_frames_reports_ffmpeg_failure(tmp_path):
    # Регрессия: сбой ffmpeg при нарезке кадров раньше проходил молча
    # и выглядел как «ffmpeg не вернул ни одного кадра».
    with pytest.raises(MediaProcessingError):
        frames_mod.extract_frames(tmp_path / "missing.mp4", tmp_path / "frames")
