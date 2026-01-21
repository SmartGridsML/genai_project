import asyncio
import base64
import logging
import time
from typing import Any, Dict, Optional

import httpx

from backend.app.celery_app import celery_app
from backend.app.config import settings
from backend.app.core.document_parser import parse_cv
from backend.app.core.auditor import get_auditor
from backend.app.core.cv_enhancer import CVEnhancer
from backend.app.models.schemas import ExtractedFacts
from backend.app.services.cache_service import CacheService
from backend.app.services.llm_client import LLMClient
from backend.app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

STATUS_KEY_PREFIX = "application:status:"
RESULT_KEY_PREFIX = "application:result:"
LOCK_TTL_SECONDS = 120


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _set_status(cache: CacheService, request_id: str, status: str, detail: Optional[str] = None) -> None:
    payload = {
        "request_id": request_id,
        "status": status,
        "detail": detail,
        "updated_at": _now_iso(),
    }
    cache.set_json(
        f"{STATUS_KEY_PREFIX}{request_id}",
        payload,
        ttl_seconds=settings.cache_ttl_seconds,
    )


async def _get_or_compute_cached_json(
    cache: CacheService,
    key: str,
    ttl_seconds: int,
    compute_fn,
) -> Dict[str, Any]:
    cached = cache.get_json(key)
    if cached is not None:
        return cached

    lock_key = f"lock:{key}"
    acquired = cache.client.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS)
    if acquired:
        try:
            value = await compute_fn()
            cache.set_json(key, value, ttl_seconds=ttl_seconds)
            return value
        finally:
            cache.client.delete(lock_key)

    for _ in range(20):
        await asyncio.sleep(0.5)
        cached = cache.get_json(key)
        if cached is not None:
            return cached

    value = await compute_fn()
    cache.set_json(key, value, ttl_seconds=ttl_seconds)
    return value


async def _resolve_facts_and_jd(
    cache: CacheService,
    llm: LLMClient,
    parsed_sections: Dict[str, str],
    job_description: str,
    facts_cache_key: str,
    jd_cache_key: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    tasks = {}

    if cache.get_json(facts_cache_key) is None:
        tasks["facts"] = asyncio.create_task(
            _get_or_compute_cached_json(
                cache,
                facts_cache_key,
                settings.cache_ttl_seconds,
                lambda: llm.extract_facts(parsed_sections),
            )
        )

    if cache.get_json(jd_cache_key) is None:
        tasks["jd"] = asyncio.create_task(
            _get_or_compute_cached_json(
                cache,
                jd_cache_key,
                settings.cache_ttl_seconds,
                lambda: llm.analyze_jd(job_description),
            )
        )

    results = {}
    if tasks:
        done = await asyncio.gather(*tasks.values())
        for key, value in zip(tasks.keys(), done):
            results[key] = value

    facts = cache.get_json(facts_cache_key) or results.get("facts")
    jd = cache.get_json(jd_cache_key) or results.get("jd")

    if facts is None or jd is None:
        raise RuntimeError("Failed to resolve facts or job analysis")

    return facts, jd


async def _post_webhook(callback_url: str, payload: Dict[str, Any]) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(callback_url, json=payload)
    except Exception as exc:
        logger.warning("Webhook delivery failed: %s", exc)


async def _run_pipeline(payload: Dict[str, Any]) -> None:
    cache = CacheService(settings.redis_url)
    request_id = payload["request_id"]

    _set_status(cache, request_id, "processing")

    file_bytes = base64.b64decode(payload["file_b64"])
    filename = payload["filename"]
    job_description = payload["job_description"]
    tone = payload.get("tone") or "professional"
    callback_url = payload.get("callback_url")

    parsed = parse_cv(file_bytes, filename=filename)

    cv_hash = payload["cv_hash"]
    jd_hash = payload["jd_hash"]
    facts_cache_key = f"facts:{cv_hash}"
    jd_cache_key = f"jd:{jd_hash}"

    llm_client = LLMClient(settings.llm_base_url)

    facts, jd = await _resolve_facts_and_jd(
        cache,
        llm_client,
        parsed.sections,
        job_description,
        facts_cache_key,
        jd_cache_key,
    )

    cover = await llm_client.generate_cover_letter(facts=facts, jd=jd, tone=tone)
    cover_letter_text = cover.get("cover_letter", "")

    try:
        auditor = get_auditor()
        facts_model = ExtractedFacts.model_validate(facts)
        audit_report_obj = auditor.audit(
            cover_letter=cover_letter_text,
            fact_table=facts_model,
            request_id=request_id,
        )
        audit_report = audit_report_obj.model_dump()
    except Exception as exc:
        audit_report = {"error": str(exc)}

    try:
        enhancer = CVEnhancer(llm_service=LLMService())
        cv_patches = enhancer.enhance(
            original_cv_text=parsed.raw_text,
            fact_table=facts,
            jd_requirements=jd,
            max_suggestions=8,
        )
        cv_suggestions = [
            {
                "section": patch.section,
                "before": patch.before,
                "after": patch.after,
                "rationale": patch.rationale,
                "grounded_sources": patch.grounded_sources,
                "diff_unified": patch.diff_unified,
            }
            for patch in cv_patches
        ]
    except Exception as exc:
        cv_suggestions = [{"error": str(exc)}]

    result_blob = {
        "request_id": request_id,
        "filename": filename,
        "cv_hash": cv_hash,
        "jd_hash": jd_hash,
        "tone": tone,
        "cover_letter": cover_letter_text,
        "audit_report": audit_report,
        "cv_suggestions": cv_suggestions,
        "cv_raw_text": parsed.raw_text,
    }

    cache.set_json(
        f"{RESULT_KEY_PREFIX}{request_id}",
        result_blob,
        ttl_seconds=settings.cache_ttl_seconds,
    )

    _set_status(cache, request_id, "completed")

    if callback_url:
        await _post_webhook(callback_url, {
            "request_id": request_id,
            "status": "completed",
            "result": result_blob,
        })


async def process_application_async(payload: Dict[str, Any]) -> None:
    cache = CacheService(settings.redis_url)
    request_id = payload.get("request_id", "unknown")

    _set_status(cache, request_id, "queued")

    try:
        await _run_pipeline(payload)
    except Exception as exc:
        logger.exception("Failed to process application %s", request_id)
        _set_status(cache, request_id, "failed", detail=str(exc))
        callback_url = payload.get("callback_url")
        if callback_url:
            await _post_webhook(callback_url, {
                "request_id": request_id,
                "status": "failed",
                "error": str(exc),
            })
        raise


@celery_app.task(name="applications.process")
def process_application(payload: Dict[str, Any]) -> None:
    asyncio.run(process_application_async(payload))
