from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette import status
import json
import logging
import time
from uuid import uuid4
from functools import lru_cache

from backend.app.core.document_parser import parse_cv

import hashlib
from fastapi import Request, Form
from backend.app.config import settings
from backend.app.services.cache_service import CacheService
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form, Depends
from backend.app.services.llm_client import LLMClient, get_llm_client
from backend.app.core.cv_enhancer import CVEnhancer
from backend.app.services.llm_service import LLMService
from backend.app.core.auditor import get_auditor
from backend.app.models.schemas import ExtractedFacts


router = APIRouter(prefix="/applications", tags=["applications"])
logger = logging.getLogger(__name__)

def get_cache() -> CacheService:
    return CacheService(settings.redis_url)

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _log_event(payload: dict) -> None:
    logger.info(json.dumps(payload))

@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()

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
    llm: LLMClient = Depends(get_llm_client),
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
    t0 = time.perf_counter()
    cover = await llm.generate_cover_letter(facts=facts, jd=jd, tone=tone)
    _log_event({"event": "stage_complete", "stage": "cover_letter", "ms": round((time.perf_counter()-t0)*1000, 2)})

    request_id = getattr(request.state, "request_id", None) or str(uuid4())

    cover_letter_text = cover.get("cover_letter", "")

    # 5) audit report (Person A Thursday)
    # If auditor is not ready/exposed, you can temporarily set audit_report = []
    t0 = time.perf_counter()
    try:
        auditor = get_auditor()
        facts_model = ExtractedFacts.model_validate(facts)

        audit_report_obj = auditor.audit(
            cover_letter=cover_letter_text,
            fact_table=facts_model,
            request_id=request_id,
        )
        audit_report = audit_report_obj.model_dump()
    except Exception as e:
        audit_report = {"error": str(e)}


    _log_event({"event": "stage_complete", "stage": "audit", "request_id": request_id, "ms": round((time.perf_counter()-t0)*1000, 2)})

    # 6) cv enhancement
    t0 = time.perf_counter()
    try:
        enhancer = CVEnhancer(llm_service=get_llm_service())
        # Use full raw text for "before" snippets to match exactly
        original_cv_text = parsed.raw_text
        cv_patches = enhancer.enhance(
            original_cv_text=original_cv_text,
            fact_table=facts,
            jd_requirements=jd,
            max_suggestions=8,
        )
        cv_suggestions = [
            {
                "section": p.section,
                "before": p.before,
                "after": p.after,
                "rationale": p.rationale,
                "grounded_sources": p.grounded_sources,
                "diff_unified": p.diff_unified,
            }
            for p in cv_patches
        ]
    except Exception as e:
        cv_suggestions = [{"error": str(e)}]

    _log_event({"event": "stage_complete", "stage": "cv_enhance", "request_id": request_id, "ms": round((time.perf_counter()-t0)*1000, 2)})

    # 7) store results
    t0 = time.perf_counter()
    result_blob = {
        "request_id": request_id,
        "filename": file.filename,
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "tone": tone,
        "cover_letter": cover_letter_text,
        "audit_report": audit_report,
        "cv_suggestions": cv_suggestions,
        "cv_raw_text": parsed.raw_text,
    }

    cache.set_json(f"application:result:{request_id}", result_blob, ttl_seconds=settings.cache_ttl_seconds)
    _log_event({"event": "stage_complete", "stage": "store_results", "request_id": request_id, "ms": round((time.perf_counter()-t0)*1000, 2)})

    # Keep response small; fetch full payload via /applications/{id}/results
    return {
        "request_id": request_id,
        "filename": file.filename,
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "tone": tone,
    }


@router.get("/{request_id}/results")
async def get_application_results(request_id: str):
    cache = get_cache()
    data = cache.get_json(f"application:result:{request_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Results not found (expired or invalid request id)")
    return data
