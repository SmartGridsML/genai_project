import base64
import hashlib
import os
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from starlette import status

from backend.app.config import settings
from backend.app.core.document_parser import parse_cv
from backend.app.services.cache_service import CacheService
from backend.app.tasks.application_tasks import process_application, process_application_async


router = APIRouter(prefix="/applications", tags=["applications"])
STATUS_KEY_PREFIX = "application:status:"
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


def get_cache() -> CacheService:
    return CacheService(settings.redis_url)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@router.post("/parse")
async def parse_application_cv(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()

    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf and .docx files are supported.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Max size is 5MB.",
        )

    try:
        parsed = parse_cv(content, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "filename": file.filename,
        "detected_headings": parsed.detected_headings,
        "sections": parsed.sections,
        "raw_text_preview": parsed.raw_text[:1000],
    }


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_application(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(..., min_length=20),
    tone: str = Form("professional"),
    callback_url: str | None = Form(None),
):
    filename = (file.filename or "").lower()

    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf and .docx files are supported.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Max size is 5MB.",
        )

    cv_hash = _sha256(content)
    jd_hash = _sha256(job_description.encode("utf-8"))
    request_id = getattr(request.state, "request_id", None) or str(uuid4())

    cache = get_cache()
    cache.set_json(
        f"{STATUS_KEY_PREFIX}{request_id}",
        {
            "request_id": request_id,
            "status": "queued",
            "updated_at": None,
        },
        ttl_seconds=settings.cache_ttl_seconds,
    )

    payload = {
        "request_id": request_id,
        "filename": file.filename or "upload",
        "file_b64": base64.b64encode(content).decode("ascii"),
        "job_description": job_description,
        "tone": tone,
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "callback_url": callback_url,
    }
    if settings.celery_always_eager or os.getenv("PYTEST_CURRENT_TEST"):
        await process_application_async(payload)
    else:
        process_application.delay(payload)

    return {
        "request_id": request_id,
        "filename": file.filename,
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "tone": tone,
        "status": "queued",
    }


@router.get("/{request_id}/status")
async def get_application_status(request_id: str):
    cache = get_cache()
    data = cache.get_json(f"{STATUS_KEY_PREFIX}{request_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Status not found (expired or invalid request id)")
    return data


@router.get("/{request_id}/results")
async def get_application_results(request_id: str):
    cache = get_cache()
    data = cache.get_json(f"application:result:{request_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Results not found (expired or invalid request id)")
    return data
