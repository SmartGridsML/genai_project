from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette import status

from app.core.document_parser import parse_cv

import hashlib
from fastapi import Request, Form
from app.config import settings
from app.services.cache_service import CacheService
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/applications", tags=["applications"])
def get_cache() -> CacheService:
    return CacheService(settings.redis_url)

def get_llm_client() -> LLMClient:
    return LLMClient(settings.llm_base_url)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": file.filename,
        "detected_headings": parsed.detected_headings,
        "sections": parsed.sections,
        "raw_text_preview": parsed.raw_text[:1000],  # preview only; avoids huge responses
    }

@router.post("/generate")
async def generate_application(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(..., min_length=20),
    tone: str = Form("professional"),
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

    # 1) Parse CV
    try:
        parsed = parse_cv(content, filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # cache keys
    cv_hash = _sha256(content)
    jd_hash = _sha256(job_description.encode("utf-8"))
    facts_cache_key = f"facts:{cv_hash}"
    jd_cache_key = f"jd:{jd_hash}"

    # 2) facts (cached)
    cache = get_cache()
    llm = get_llm_client()
    facts = cache.get_json(facts_cache_key)
    if facts is None:
        facts = await llm.extract_facts(parsed.sections)
        cache.set_json(facts_cache_key, facts, ttl_seconds=settings.cache_ttl_seconds)

    # 3) jd analysis (cached)
    jd = cache.get_json(jd_cache_key)
    if jd is None:
        jd = await llm.analyze_jd(job_description)
        cache.set_json(jd_cache_key, jd, ttl_seconds=settings.cache_ttl_seconds)

    # 4) cover letter
    cover = await llm.generate_cover_letter(facts=facts, jd=jd, tone=tone)

    request_id = getattr(request.state, "request_id", None)

    return {
        "request_id": request_id,
        "filename": file.filename,
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "tone": tone,
        "cover_letter": cover.get("cover_letter", ""),
        "facts": facts,  # keep for debugging; you can remove later
        "jd": jd,        # keep for debugging; you can remove later
    }
