import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, HttpUrl

from src import pipeline
from src.config import MAX_UPLOAD_BYTES, MAX_VIDEO_SECONDS
from src.errors import AppError
from src.models import CONTRACT_VERSION, Recipe

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fropti Video-to-Recipe API",
    version="0.1.0",
    description=(
        f"Контракт Recipe {CONTRACT_VERSION}. Принимает URL YouTube/Shorts или загруженный "
        f"файл длительностью до {MAX_VIDEO_SECONDS} секунд и возвращает валидированный Recipe."
    ),
)

_ERROR_RESPONSES = {
    400: {"description": "empty_file, video_too_long, unsupported_platform, ssrf_blocked, video_unreadable"},
    413: {"description": "file_too_large"},
    422: {"description": "no_usable_content или стандартная ошибка валидации тела"},
    500: {"description": "media_processing_failed, ocr_failed, internal_error"},
    502: {"description": "download_failed, llm_unavailable, llm_invalid_response"},
    503: {"description": "model_unavailable"},
}


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "contract_version": CONTRACT_VERSION}


@app.post("/extract", response_model=Recipe, responses=_ERROR_RESPONSES)
def extract_recipe(request: ExtractRequest) -> Recipe:
    return _extract(str(request.url))


@app.post("/extract/upload", response_model=Recipe, responses=_ERROR_RESPONSES)
def extract_uploaded_recipe(
    file: Annotated[UploadFile, File(json_schema_extra={"format": "binary"})],
) -> Recipe:
    if not file.size:
        raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "Файл пуст"})
    if file.size > MAX_UPLOAD_BYTES:
        limit_mib = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            # :g печатает 100, а не 100.0, и корректно показывает дробные лимиты (7.5).
            detail={"code": "file_too_large", "message": f"Файл должен быть не больше {limit_mib:g} MiB"},
        )
    with TemporaryDirectory(prefix="fropti-") as directory:
        video_path = Path(directory) / "video"
        with video_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination, length=1024 * 1024)
        return _extract(video_path)


def _extract(source: str | Path) -> Recipe:
    # AppError несёт HTTP-статус и машинный код; непредвиденные ошибки — 500.
    try:
        return pipeline.extract_recipe(source)
    except AppError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        # Не-AppError (сбой загрузки модели, баг) тоже уходит клиенту как JSON;
        # подробности — только в логах.
        logger.exception("Непредвиденная ошибка при извлечении рецепта")
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": "Внутренняя ошибка сервиса; подробности в логах"},
        ) from exc
