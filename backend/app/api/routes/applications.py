import asyncio
import hashlib
import json
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from starlette import status

from backend.app.config import settings
from backend.app.core.auditor import Auditor
from backend.app.core.cv_enhancer import CVEnhancer
from backend.app.core.document_parser import parse_cv
from backend.app.models.schemas import ApplicationGenerateRequest, ExtractedFacts
from backend.app.services.cache_service import CacheService
from backend.app.services.llm_client import LLMClient, get_llm_client
from backend.app.utils.rate_limiter import limiter
from backend.app.utils.result_token import sign as sign_token, verify as verify_token

router = APIRouter(prefix="/applications", tags=["applications"])
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
JOB_QUEUE_KEY = "application:jobs:queue"
JOB_PROCESSING_KEY = "application:jobs:processing"
WORKER_BRPOP_TIMEOUT_SECONDS = 1

_worker_task: asyncio.Task | None = None
_worker_stop_event: asyncio.Event | None = None


def get_cache() -> CacheService:
    return CacheService()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _log_event(payload: dict) -> None:
    logger.info(json.dumps(payload))


async def _load_result_from_cache(cache: CacheService, request_id: str) -> dict | None:
    key = f"application:result:{request_id}"
    try:
        data = await cache.get_encrypted_json(key)
    except Exception:
        data = None
    if data is None:
        data = await cache.get_json(key)
    return data


async def _enqueue_generation_job(
    cache: CacheService,
    request_id: str,
    body: ApplicationGenerateRequest,
) -> None:
    job_payload = {"request_id": request_id, "body": body.model_dump()}
    await cache.client.lpush(JOB_QUEUE_KEY, json.dumps(job_payload).encode("utf-8"))


async def _requeue_stale_jobs(cache: CacheService) -> int:
    moved = 0
    while True:
        payload = await cache.client.rpop(JOB_PROCESSING_KEY)
        if payload is None:
            break
        await cache.client.lpush(JOB_QUEUE_KEY, payload)
        moved += 1
    return moved


async def _generation_worker_loop(stop_event: asyncio.Event) -> None:
    cache = get_cache()
    logger.info("Generation worker loop running")

    while not stop_event.is_set():
        payload = await cache.client.brpoplpush(
            JOB_QUEUE_KEY,
            JOB_PROCESSING_KEY,
            timeout=WORKER_BRPOP_TIMEOUT_SECONDS,
        )
        if payload is None:
            continue

        payload_text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        request_id = None
        try:
            parsed = json.loads(payload_text)
            request_id = str(parsed["request_id"])
            body = ApplicationGenerateRequest.model_validate(parsed["body"])

            existing = await _load_result_from_cache(cache, request_id)
            if isinstance(existing, dict) and existing.get("status") in {"done", "failed"}:
                logger.info(f"Skipping already completed request_id={request_id}")
                continue

            llm = get_llm_client()
            await _run_generation_pipeline(request_id=request_id, body=body, llm=llm)
        except Exception as exc:
            logger.error(f"Worker failed processing payload: {exc}", exc_info=True)
            if request_id:
                await cache.set_json(
                    f"application:result:{request_id}",
                    {"request_id": request_id, "status": "failed", "error": str(exc)},
                    ttl_seconds=settings.cache_ttl_seconds,
                )
        finally:
            await cache.client.lrem(JOB_PROCESSING_KEY, 1, payload)


async def start_generation_worker() -> None:
    global _worker_task, _worker_stop_event

    if _worker_task is not None and not _worker_task.done():
        return

    cache = get_cache()
    moved = await _requeue_stale_jobs(cache)
    if moved:
        logger.warning(f"Requeued {moved} stale generation jobs from processing list")

    _worker_stop_event = asyncio.Event()
    _worker_task = asyncio.create_task(_generation_worker_loop(_worker_stop_event), name="generation-worker")
    logger.info("Generation worker started")


async def stop_generation_worker() -> None:
    global _worker_task, _worker_stop_event

    if _worker_task is None:
        return

    if _worker_stop_event is not None:
        _worker_stop_event.set()

    try:
        await asyncio.wait_for(_worker_task, timeout=30)
    except asyncio.TimeoutError:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    finally:
        _worker_task = None
        _worker_stop_event = None
        logger.info("Generation worker stopped")


@router.post("/parse")
@limiter.limit("30/minute")
async def parse_application_cv(request: Request, file: UploadFile = File(...)):
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
        "parsed_text": parsed.raw_text,
        "detected_headings": parsed.detected_headings,
        "sections": parsed.sections,
    }


@router.post("/generate")
@limiter.limit("10/minute")
async def generate_application(
    request: Request,
    body: ApplicationGenerateRequest,
):
    request_id = getattr(request.state, "request_id", None) or str(uuid4())
    access_token = sign_token(request_id, ttl_seconds=settings.cache_ttl_seconds)
    cv_hash = _sha256(body.cv_text.encode("utf-8"))
    jd_hash = _sha256(body.job_description.encode("utf-8"))
    cache = get_cache()

    await cache.set_json(
        f"application:result:{request_id}",
        {"request_id": request_id, "status": "processing"},
        ttl_seconds=settings.cache_ttl_seconds,
    )

    try:
        await _enqueue_generation_job(cache, request_id, body)
    except Exception as exc:
        logger.error(f"Failed to enqueue generation job for {request_id}: {exc}", exc_info=True)
        await cache.set_json(
            f"application:result:{request_id}",
            {"request_id": request_id, "status": "failed", "error": "Could not queue generation job"},
            ttl_seconds=settings.cache_ttl_seconds,
        )
        raise HTTPException(status_code=503, detail="Could not queue generation job")

    return {
        "request_id": request_id,
        "access_token": access_token,
        "status": "processing",
        "filename": "cv_text.txt",
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "tone": body.tone or "professional",
    }


async def _run_generation_pipeline(
    request_id: str,
    body: ApplicationGenerateRequest,
    llm: LLMClient,
) -> None:
    cache = get_cache()
    cv_text = body.cv_text
    job_description = body.job_description
    tone = body.tone or "professional"
    cv_hash = _sha256(cv_text.encode("utf-8"))
    jd_hash = _sha256(job_description.encode("utf-8"))
    sections = {"raw": cv_text}

    try:
        async def _fetch_facts() -> dict:
            key = f"facts:{cv_hash}"
            try:
                cached = await cache.get_encrypted_json(key)
            except Exception:
                cached = await cache.get_json(key)
            if cached is not None:
                return cached
            result = await llm.extract_facts(sections)
            await cache.set_encrypted_json(key, result, ttl_seconds=settings.cache_ttl_seconds)
            return result

        async def _fetch_jd() -> dict:
            key = f"jd:{jd_hash}"
            try:
                cached = await cache.get_encrypted_json(key)
            except Exception:
                cached = await cache.get_json(key)
            if cached is not None:
                return cached
            result = await llm.analyze_jd(job_description)
            await cache.set_encrypted_json(key, result, ttl_seconds=settings.cache_ttl_seconds)
            return result

        t0 = time.perf_counter()
        facts, jd = await asyncio.gather(_fetch_facts(), _fetch_jd())
        _log_event(
            {
                "event": "stage_complete",
                "stage": "facts_and_jd",
                "request_id": request_id,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )

        t0 = time.perf_counter()
        cover = await llm.generate_cover_letter(facts=facts, jd=jd, tone=tone)
        cover_letter_text = cover.get("cover_letter", "")
        _log_event(
            {
                "event": "stage_complete",
                "stage": "cover_letter",
                "request_id": request_id,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )

        async def _run_audit() -> dict:
            try:
                auditor = Auditor(llm_client=llm)
                facts_model = ExtractedFacts.model_validate(facts)
                report = await auditor.audit(
                    cover_letter=cover_letter_text,
                    fact_table=facts_model,
                    request_id=request_id,
                )
                return report.model_dump()
            except Exception as exc:
                logger.error(f"Audit failed for request {request_id}: {exc}", exc_info=True)
                return {"error": str(exc)}

        async def _run_cv_enhance() -> list:
            try:
                enhancer = CVEnhancer(llm_client=llm)
                patches = await enhancer.enhance(
                    original_cv_text=cv_text,
                    fact_table=facts,
                    jd_requirements=jd,
                    max_suggestions=8,
                )
                return [
                    {
                        "section": p.section,
                        "before": p.before,
                        "after": p.after,
                        "rationale": p.rationale,
                        "grounded_sources": p.grounded_sources,
                        "diff_unified": p.diff_unified,
                    }
                    for p in patches
                ]
            except Exception as exc:
                logger.error(f"CV enhancement failed for request {request_id}: {exc}", exc_info=True)
                return [{"error": str(exc)}]

        t0 = time.perf_counter()
        audit_report, cv_suggestions = await asyncio.gather(_run_audit(), _run_cv_enhance())
        _log_event(
            {
                "event": "stage_complete",
                "stage": "audit_and_enhance",
                "request_id": request_id,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )

        warnings = []
        if isinstance(audit_report, dict) and "error" in audit_report:
            warnings.append(f"Audit failed: {audit_report['error']}")
        if cv_suggestions and isinstance(cv_suggestions[0], dict) and "error" in cv_suggestions[0]:
            warnings.append(f"CV enhancement failed: {cv_suggestions[0]['error']}")

        result_blob = {
            "request_id": request_id,
            "status": "done",
            "warnings": warnings,
            "filename": "cv_text.txt",
            "cv_hash": cv_hash,
            "jd_hash": jd_hash,
            "tone": tone,
            "cover_letter": cover_letter_text,
            "audit_report": audit_report,
            "cv_suggestions": cv_suggestions,
            "cv_raw_text": cv_text,
        }

        await cache.set_encrypted_json(
            f"application:result:{request_id}",
            result_blob,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        _log_event({"event": "stage_complete", "stage": "store_results", "request_id": request_id})

    except Exception as exc:
        logger.error(f"Generation pipeline failed for {request_id}: {exc}", exc_info=True)
        await cache.set_json(
            f"application:result:{request_id}",
            {"request_id": request_id, "status": "failed", "error": str(exc)},
            ttl_seconds=settings.cache_ttl_seconds,
        )


@router.get("/{request_id}/results")
async def get_application_results(
    request_id: str,
    authorization: str | None = Header(default=None),
    cache: CacheService = Depends(get_cache),
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'")

    token = authorization.removeprefix("Bearer ")
    if not verify_token(request_id, token):
        raise HTTPException(status_code=403, detail="Invalid access token")

    data = await _load_result_from_cache(cache, request_id)
    if not data:
        raise HTTPException(status_code=404, detail="Results not found (expired or invalid request id)")
    return data
