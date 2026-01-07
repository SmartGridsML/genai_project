from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FactExtractRequest(BaseModel):
    sections: Dict[str, str] = Field(..., description="CV sections from parser")


class FactExtractResponse(BaseModel):
    experiences: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    raw_sections: Optional[Dict[str, str]] = None


class JDAnalyzeRequest(BaseModel):
    job_description: str = Field(..., min_length=20)


class JDAnalyzeResponse(BaseModel):
    requirements: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class CoverLetterRequest(BaseModel):
    facts: Dict[str, Any]
    job: Dict[str, Any]  # IMPORTANT: matches LLMClient payload
    tone: str = "professional"


class CoverLetterResponse(BaseModel):
    cover_letter: str
    tone: str = "professional"
