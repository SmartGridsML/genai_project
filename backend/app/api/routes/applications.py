from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette import status

from app.core.document_parser import parse_cv

router = APIRouter(prefix="/applications", tags=["applications"])

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
