from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import Response

from backend.app.services.cache_service import CacheService
from backend.app.services.document_service import DocumentService
from backend.app.utils.result_token import verify as verify_token

router = APIRouter(prefix="/applications", tags=["applications"])


def get_cache() -> CacheService:
    return CacheService()


def get_docs() -> DocumentService:
    return DocumentService()


async def _load_result(cache: CacheService, request_id: str, authorization: str | None) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be 'Bearer <token>'")

    access_token = authorization.removeprefix("Bearer ")
    if not verify_token(request_id, access_token):
        raise HTTPException(status_code=403, detail="Invalid access token")

    try:
        data = await cache.get_encrypted_json(f"application:result:{request_id}")
    except Exception:
        data = await cache.get_json(f"application:result:{request_id}")

    if not data:
        raise HTTPException(status_code=404, detail="Results not found (expired or invalid request id)")

    result_status = data.get("status")
    if result_status == "processing":
        raise HTTPException(status_code=409, detail="Result is still processing")
    if result_status == "failed":
        raise HTTPException(status_code=409, detail=data.get("error", "Generation failed"))
    if result_status != "done":
        raise HTTPException(status_code=409, detail=f"Result is not downloadable (status={result_status})")

    return data


@router.get("/{request_id}/download/cover-letter.docx")
async def download_cover_letter_docx(
    request_id: str,
    authorization: str | None = Header(default=None),
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = await _load_result(cache, request_id, authorization)
    cover = data.get("cover_letter", "") or ""
    f = docs.cover_letter_docx(cover, filename="cover_letter.docx")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


@router.get("/{request_id}/download/cover-letter.pdf")
async def download_cover_letter_pdf(
    request_id: str,
    authorization: str | None = Header(default=None),
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = await _load_result(cache, request_id, authorization)
    cover = data.get("cover_letter", "") or ""
    f = docs.cover_letter_pdf(cover, filename="cover_letter.pdf")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


@router.get("/{request_id}/download/enhanced-cv.docx")
async def download_enhanced_cv_docx(
    request_id: str,
    authorization: str | None = Header(default=None),
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = await _load_result(cache, request_id, authorization)
    cv_text = data.get("cv_raw_text", "") or ""
    suggestions = data.get("cv_suggestions", []) or []
    bullets = _suggestions_to_bullets(suggestions)
    f = docs.enhanced_cv_docx(cv_text, bullets, filename="enhanced_cv.docx")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


@router.get("/{request_id}/download/enhanced-cv.pdf")
async def download_enhanced_cv_pdf(
    request_id: str,
    authorization: str | None = Header(default=None),
    cache: CacheService = Depends(get_cache),
    docs: DocumentService = Depends(get_docs),
):
    data = await _load_result(cache, request_id, authorization)
    cv_text = data.get("cv_raw_text", "") or ""
    suggestions = data.get("cv_suggestions", []) or []
    bullets = _suggestions_to_bullets(suggestions)
    f = docs.enhanced_cv_pdf(cv_text, bullets, filename="enhanced_cv.pdf")
    return Response(
        content=f.data,
        media_type=f.content_type,
        headers={"Content-Disposition": f'attachment; filename="{f.filename}"'},
    )


def _suggestions_to_bullets(suggestions: list) -> list:
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
    return bullets
