from __future__ import annotations

from fastapi import APIRouter

from app.models.llm_contracts import (
    FactExtractRequest,
    FactExtractResponse,
    JDAnalyzeRequest,
    JDAnalyzeResponse,
    CoverLetterRequest,
    CoverLetterResponse,
)
from app.core.jd_analyzer import analyze_job_description
from app.core.cover_letter_gen import generate_cover_letter
from app.services.llm_service import get_llm_service

router = APIRouter(tags=["llm"])


@router.post("/fact-extract", response_model=FactExtractResponse)
async def fact_extract(req: FactExtractRequest):
    # Stub until Day 2 extractor lands
    return FactExtractResponse(raw_sections=req.sections)


@router.post("/jd-analyze", response_model=JDAnalyzeResponse)
async def jd_analyze(req: JDAnalyzeRequest):
    llm = get_llm_service()
    out = await analyze_job_description(llm, req.job_description)
    return JDAnalyzeResponse(**out)


@router.post("/cover-letter", response_model=CoverLetterResponse)
async def cover_letter(req: CoverLetterRequest):
    llm = get_llm_service()
    letter = await generate_cover_letter(llm_service=llm, facts=req.facts, job=req.job, tone=req.tone)
    return CoverLetterResponse(cover_letter=letter, tone=req.tone)
