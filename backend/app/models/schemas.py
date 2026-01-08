from pydantic import BaseModel, Field
from typing import Optional, List


class HealthResponse(BaseModel):
    status: str = "ok"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# Placeholder for Day-2/3 endpoints
class ApplicationGenerateRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    tone: Optional[str] = Field(default="professional")
    # later: cv_file upload handled via multipart/form-data


class ApplicationGenerateResponse(BaseModel):
    request_id: str
    cover_letter: str
    cv_suggestions: List[str] = []


class KeyFact(BaseModel):
    category: str = Field(..., description="")
    fact: str = Field(..., description="")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class ExtractedFacts(BaseModel):
    facts: List[KeyFact]


class JobAnalysis(BaseModel):
    summary: str
    required_skills: List[str]
    experience_level: str
    remote_policy: str