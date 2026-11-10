from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

from fropti_core_contracts import UploadSource, UrlSource

from src import downloader
from src.config import MAX_VIDEO_SECONDS
from src.errors import AppError, VideoUnreadableError
from src.extractors import frames as frames_mod
from src.extractors import ocr as ocr_mod
from src.extractors import stt as stt_mod
from src.llm import fusion
from src.models import Recipe


class VideoTooLongError(AppError):
    code = "video_too_long"
    http_status = 400


def extract_recipe(source: str | Path) -> Recipe:
    # 1. Длительность проверяется до любой тяжёлой обработки.
    duration = downloader.probe_duration(source)
    if not math.isfinite(duration) or duration <= 0:
        raise VideoUnreadableError(f"Не удалось определить корректную длительность видео: {duration}")
    if duration > MAX_VIDEO_SECONDS:
        raise VideoTooLongError(
            f"Видео должно быть не длиннее {MAX_VIDEO_SECONDS} секунд; получено {duration:.1f}"
        )
    # 2. Локальный файл обрабатываем на месте; URL скачиваем во временную папку.
    with TemporaryDirectory(prefix="fropti-pipeline-") as directory:
        work_dir = Path(directory)
        if isinstance(source, Path):
            video_path = source
            recipe_source = UploadSource(type="upload")
        else:
            video_path = downloader.download_video(source, work_dir)
            recipe_source = UrlSource(type="url", url=source)
        # 3. Факты: речь (STT) и текст с экрана (кадры -> OCR).
        transcript = stt_mod.transcribe(video_path)
        frame_paths = frames_mod.extract_frames(video_path, work_dir / "frames")
        screen_text = ocr_mod.recognize_text(frame_paths)
        # 4. LLM собирает Recipe; source берётся из реального входа, а не из LLM.
        return fusion.fuse_recipe(transcript, screen_text, recipe_source)
