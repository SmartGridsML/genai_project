from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.app.config import settings
from backend.app.services.cache_service import CacheService
from backend.app.services.document_service import DocumentService

router = APIRouter(prefix="/applications", tags=["applications"])


def get_cache() -> CacheService:
    return CacheService(settings.redis_url)


def get_docs() -> DocumentService:
    return DocumentService()


def _load_result(cache: CacheService, request_id: str) -> dict:
    data = cache.get_json(f"application:result:{request_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Results not found (expired or invalid request id)")
    return data


@router.get("/{request_id}/download/cover-letter.docx")
def download_cover_letter_docx(
    request_id: str,
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = _load_result(cache, request_id)
    cover = data.get("cover_letter", "") or ""
    f = docs.cover_letter_docx(cover, filename="cover_letter.docx")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


@router.get("/{request_id}/download/cover-letter.pdf")
def download_cover_letter_pdf(
    request_id: str,
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = _load_result(cache, request_id)
    cover = data.get("cover_letter", "") or ""
    f = docs.cover_letter_pdf(cover, filename="cover_letter.pdf")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


@router.get("/{request_id}/download/enhanced-cv.docx")
def download_enhanced_cv_docx(
    request_id: str,
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = _load_result(cache, request_id)
    cv_text = data.get("cv_raw_text", "") or ""

    # your cv_suggestions are dicts; turn them into readable bullet strings
    suggestions = data.get("cv_suggestions", []) or []
    bullets = []
    for s in suggestions:
        if isinstance(s, dict) and "error" in s:
            continue
        if isinstance(s, dict):
            after = (s.get("after") or "").strip()
            rationale = (s.get("rationale") or "").strip()
            if after and rationale:
                bullets.append(f"{after} — {rationale}")
            elif after:
                bullets.append(after)
            else:
                # last-resort stringify
                bullets.append(str(s))
        else:
            bullets.append(str(s))

    f = docs.enhanced_cv_docx(cv_text, bullets, filename="enhanced_cv.docx")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


@router.get("/{request_id}/download/enhanced-cv.pdf")
def download_enhanced_cv_pdf(
    request_id: str,
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = _load_result(cache, request_id)
    cv_text = data.get("cv_raw_text", "") or ""

    suggestions = data.get("cv_suggestions", []) or []
    bullets = []
    for s in suggestions:
        if isinstance(s, dict) and "error" in s:
            continue
        if isinstance(s, dict):
            after = (s.get("after") or "").strip()
            rationale = (s.get("rationale") or "").strip()
            if after and rationale:
                bullets.append(f"{after} — {rationale}")
            elif after:
                bullets.append(after)
            else:
                bullets.append(str(s))
        else:
            bullets.append(str(s))

    f = docs.enhanced_cv_pdf(cv_text, bullets, filename="enhanced_cv.pdf")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )
