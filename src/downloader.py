# Безопасное получение видео: длительность до тяжёлой обработки и скачивание
# только для URL из разрешённого списка платформ. Защита от SSRF: разбор URL,
# allowlist хостов, разрешение DNS с запретом частных адресов, таймауты,
# ограничение размера файла, временные файлы с очисткой на уровне пайплайна.
# Известные ограничения защиты — docs/architecture.md, раздел «Безопасность».
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from src import media
from src.config import (
    ALLOWED_URL_HOSTS,
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_DOWNLOAD_BYTES,
    MAX_VIDEO_SECONDS,
    YTDLP_COOKIES_FILE,
    YTDLP_COOKIES_FROM_BROWSER,
    YTDLP_EXTRACTOR_ARGS,
    YTDLP_JS_RUNTIME,
    YTDLP_PLAYER_CLIENT,
)
from src.errors import AppError, MediaProcessingError, VideoUnreadableError


class DownloadError(AppError):
    code = "download_failed"
    http_status = 502


class UnsupportedPlatformError(DownloadError):
    code = "unsupported_platform"
    http_status = 400


class SSRFBlockedError(DownloadError):
    code = "ssrf_blocked"
    http_status = 400


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(f"Команда {command[0]!r} превысила лимит времени {timeout} с") from exc
    except OSError as exc:
        raise DownloadError(f"Не удалось запустить {command[0]!r}: {exc}") from exc


# Первая строка Netscape cookies.txt должна содержать «# Netscape HTTP Cookie
# File» или «# HTTP Cookie File», иначе yt-dlp падает с невнятным traceback.
_NETSCAPE_MAGIC = re.compile(r"#( Netscape)? HTTP Cookie File", re.IGNORECASE)


def _assert_cookies_readable() -> None:
    # Сломанный cookiefile роняет yt-dlp по-разному, а в API это выглядит как
    # необъяснимый сбой, поэтому всё проверяем заранее:
    #   1) каталог вместо файла — Docker при раскомментированном монтировании
    #      и отсутствии файла на хосте создаёт пустый каталог;
    #   2) файл отсутствует или нечитаем — cookies молча не работают;
    #   3) не Netscape-формат (пустой, JSON-экспорт, UTF-16) — невнятный
    #      LoadError/UnicodeDecodeError;
    #   4) только чтение («:ro», docker secrets) — yt-dlp перезаписывает файл
    #      при выходе даже после неудачного извлечения → PermissionError.
    if not YTDLP_COOKIES_FILE:
        return
    path = Path(YTDLP_COOKIES_FILE)
    if path.is_dir():
        raise DownloadError(
            f"По пути {YTDLP_COOKIES_FILE!r} — каталог, а не файл cookies. Так бывает, "
            "когда в compose-файле раскомментировано монтирование «./cookies.txt:/cookies.txt», "
            "но самого файла cookies.txt рядом с compose-файлом ещё нет: Docker вместо "
            "ошибки создаёт пустой каталог. Удалите этот каталог, экспортируйте cookies.txt "
            "(расширение «Get cookies.txt LOCALLY») рядом с compose-файлом и пересоздайте "
            "контейнер: docker compose up -d."
        )
    if not path.exists() or not os.access(YTDLP_COOKIES_FILE, os.R_OK):
        raise DownloadError(
            f"Файл cookies не найден или не читается: {YTDLP_COOKIES_FILE!r}. "
            "Переменная YTDLP_COOKIES_FILE должна указывать на путь ВНУТРИ контейнера: "
            "раскомментируйте монтирование cookies.txt в compose-файле или отключите переменную."
        )
    try:
        with open(YTDLP_COOKIES_FILE, "rb") as fh:
            first_line = fh.readline(256)
    except OSError:
        return  # редкий race (файл исчез между проверками) — пусть падает yt-dlp
    try:
        header = first_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DownloadError(
            f"Файл cookies не в UTF-8 (обычно это UTF-16 от сохранения в PowerShell): "
            f"{YTDLP_COOKIES_FILE!r}. Экспортируйте cookies.txt заново расширением "
            "«Get cookies.txt LOCALLY» — первая строка файла должна быть "
            "«# Netscape HTTP Cookie File»."
        ) from exc
    if not _NETSCAPE_MAGIC.search(header):
        raise DownloadError(
            f"Файл cookies не распознан как Netscape cookies.txt: {YTDLP_COOKIES_FILE!r}. "
            "Первая строка должна начинаться с «# Netscape HTTP Cookie File» — частые "
            "причины: файл пуст, создан вручную (New-Item) или экспортирован в JSON. "
            "Нужен экспорт расширением «Get cookies.txt LOCALLY» в Netscape-формате."
        )
    if not os.access(YTDLP_COOKIES_FILE, os.W_OK):
        raise DownloadError(
            f"Файл cookies доступен только для чтения: {YTDLP_COOKIES_FILE!r}. yt-dlp "
            "перезаписывает файл при каждом запуске, поэтому монтирование с флагом «:ro» "
            "(как и /run/secrets) роняет запуск с traceback «PermissionError» в самом "
            "конце. Смонтируйте файл БЕЗ «:ro»: ./cookies.txt:/cookies.txt."
        )


def _hostname(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsupportedPlatformError(f"Поддерживаются только HTTP/HTTPS-ссылки на видео: {url!r}")
    return parsed.hostname.lower()


def _assert_host_allowed(hostname: str) -> None:
    if hostname not in ALLOWED_URL_HOSTS:
        raise UnsupportedPlatformError(
            f"Платформа {hostname!r} не поддерживается; разрешены: {', '.join(ALLOWED_URL_HOSTS)}"
        )


def _assert_public_host(hostname: str) -> None:
    # DNS разрешается здесь один раз, а yt-dlp при скачивании пере-разрешит
    # хост сам: злонамеренный DNS может вернуть разным запросам разные адреса
    # (класс «DNS-rebinding race»). Для MVP принято: хосты ограничены
    # allowlist'ом, полное закрытие требует сетевого контроля исходящих
    # соединений. Туда же относится окно редиректов yt-dlp — docs/architecture.md.
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise DownloadError(f"Не удалось разрешить адрес {hostname!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise SSRFBlockedError(f"Хост {hostname!r} указывает на запрещённый адрес {ip}")


def _stderr_excerpt(stderr: str, limit: int = 400) -> str:
    # WARNING'и yt-dlp печатает до финального ERROR, поэтому показываем конец
    # потока ошибок; сырой Python-traceback расширяет лимит — хвост traceback
    # без последней строки-исключения бесполезен для диагноза.
    text = stderr.strip()
    if "Traceback (most recent call last)" in text:
        limit = 3000
    if len(text) <= limit:
        return text
    return text[-limit:]


def _yt_dlp_command(*extra: str) -> list[str]:
    # YouTube с 2025 отдаёт плеер с challenge-скриптами: yt-dlp исполняет их в
    # JS-рантайме (node/deno/bun), без флага ищется только deno — извлечение
    # падает (см. YTDLP_JS_RUNTIME). Cookies и смена клиента плеера — против
    # проверки «Sign in to confirm you're not a bot»; оба вызова yt-dlp
    # получают их одинаково, проверка срабатывает уже на метаданных (-J).
    command = ["yt-dlp", "--no-playlist"]
    if YTDLP_JS_RUNTIME:
        command += ["--js-runtimes", YTDLP_JS_RUNTIME]
    if YTDLP_COOKIES_FILE:
        command += ["--cookies", YTDLP_COOKIES_FILE]
    if YTDLP_COOKIES_FROM_BROWSER:
        command += ["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER]
    if YTDLP_PLAYER_CLIENT:
        command += ["--extractor-args", f"youtube:player_client={YTDLP_PLAYER_CLIENT}"]
    # Произвольные extractor-args (разделитель «;»): обходной путь для меняющихся
    # требований YouTube без правки кода, например PO-токен:
    # YTDLP_EXTRACTOR_ARGS=youtube:po_token=mweb.gvs+XXX
    if YTDLP_EXTRACTOR_ARGS:
        for piece in YTDLP_EXTRACTOR_ARGS.split(";"):
            piece = piece.strip()
            if piece:
                command += ["--extractor-args", piece]
    return command + list(extra)


def _yt_dlp_metadata(url: str) -> dict:
    result = _run(_yt_dlp_command("--skip-download", "-J", url), DOWNLOAD_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise DownloadError(f"yt-dlp не получил метаданные: {_stderr_excerpt(result.stderr)}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DownloadError("yt-dlp вернул не-JSON ответ") from exc


def probe_duration(source: str | Path) -> float:
    # Длительность до STT/OCR: локальный файл — ffprobe, URL — метаданные
    # платформы (yt-dlp -J) без скачивания. Метаданные при скачивании не
    # переиспользуются — см. комментарий в download_video.
    if isinstance(source, Path):
        try:
            return media.ffprobe_duration(source)
        except media.ToolNotFoundError as exc:
            # Отсутствующий ffprobe — проблема окружения, а не файла: 5xx, а не 400.
            raise MediaProcessingError(str(exc)) from exc
        except RuntimeError as exc:
            raise VideoUnreadableError(str(exc)) from exc
    hostname = _hostname(source)
    _assert_host_allowed(hostname)
    _assert_public_host(hostname)
    _assert_cookies_readable()
    metadata = _yt_dlp_metadata(source)
    duration = metadata.get("duration")
    if duration is None:
        # Платформа ответила, но длительность не сообщила: лимит подтвердить
        # нельзя, сетевого сбоя не было — 4xx, а не 502.
        raise VideoUnreadableError(
            "Платформа не сообщила длительность видео — нельзя подтвердить, "
            f"что оно не длиннее {MAX_VIDEO_SECONDS} секунд"
        )
    try:
        return float(duration)
    except (TypeError, ValueError) as exc:
        # Мусор вместо числа не должен превращаться в необработанный ValueError.
        raise VideoUnreadableError(f"Платформа сообщила недостоверную длительность: {duration!r}") from exc


# Потолок 720p — предпочтительный, не жёсткий (PaddleOCR на 144–360p почти
# слеп): height-фильтр стоит не во всех ветках, хвост цепочки срабатывает,
# только когда у источника нет вариантов ≤720p вовсе — 1080p лучше провала
# загрузки, а размер всё равно ограничивает --max-filesize. На YouTube
# варианты ≤720p есть всегда, там потолок фактический.
# bv*+ba — дефолтная пара yt-dlp: у YouTube комбинированный файл обычно один
# и на 360p, а video-only есть до 1080p; b-ветки нужны источникам без отдельных
# аудиодорожек (ba отбирает только audio-only). worst — последний запас,
# --max-filesize считается на каждый файл до склейки.
_FORMAT_SELECTOR = (
    "bv*[height<=720][ext=mp4]+ba/bv*[height<=720]+ba"
    "/b[height<=720][ext=mp4]/b[height<=720]"
    "/bv*[ext=mp4]+ba/bv*+ba/b[ext=mp4]/b/worst"
)


def download_video(url: str, output_dir: Path) -> Path:
    # Второй вызов yt-dlp для той же страницы: лимит длительности обязан
    # срабатывать до скачивания, а подавать метаданные проверки назад через
    # --load-info-json рискованно — прямые ссылки на медиа живут минуты и к
    # моменту скачивания протухнут. Повторное извлечение стоит секунды на
    # фоне STT/OCR, измеряемых минутами.
    hostname = _hostname(url)
    _assert_host_allowed(hostname)
    _assert_public_host(hostname)
    _assert_cookies_readable()
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "source.%(ext)s")
    result = _run(
        _yt_dlp_command(
            "--restrict-filenames",
            "--max-filesize", str(MAX_DOWNLOAD_BYTES),
            "-f", _FORMAT_SELECTOR,
            "-o", template,
            url,
        ),
        DOWNLOAD_TIMEOUT_SECONDS * 4,
    )
    if result.returncode != 0:
        raise DownloadError(f"Не удалось скачать видео: {_stderr_excerpt(result.stderr)}")
    candidates = sorted(output_dir.glob("source.*"))
    if not candidates:
        raise DownloadError("После скачивания файл не появился")
    return candidates[0]
