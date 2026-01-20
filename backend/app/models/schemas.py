from pydantic import BaseModel, Field
from typing import Optional, List


class CVSuggestion(BaseModel):
    section: str = Field(..., description="CV section, e.g. Experience, Skills, Projects, Summary")
    before: str = Field(..., description="Exact snippet copied from the original CV text")
    after: str = Field(..., description="Improved snippet grounded in fact_table")
    rationale: str = Field(..., description="Why this helps match the JD")
    grounded_sources: List[str] = Field(default_factory=list, description="Pointers into fact_table items supporting the suggestion")

class CVEnhancement(BaseModel):
    suggestions: List[CVSuggestion] = Field(default_factory=list)


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
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )


class ExtractedFacts(BaseModel):
    facts: List[KeyFact]


class JobAnalysis(BaseModel):
    summary: str
    required_skills: List[str]
    experience_level: str
    remote_policy: str


class ClaimVerification(BaseModel):
    """Result of verifying a single claim against the fact table."""
    claim: str = Field(
        ..., description="The claim extracted from the cover letter"
    )
    supported: bool = Field(
        ..., description="Whether the claim is supported by the fact table"
    )
    source: str = Field(
        ..., description="Source from CV or 'UNSUPPORTED'"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score"
    )
    reasoning: Optional[str] = Field(
        None, description="Explanation of the verification decision"
    )


class AuditReport(BaseModel):
    """Complete audit report for a generated cover letter."""
    verifications: List[ClaimVerification] = Field(
        ..., description="Individual claim verifications"
    )
    total_claims: int = Field(
        ..., description="Total number of claims extracted"
    )
    supported_claims: int = Field(
        ..., description="Number of supported claims"
    )
    unsupported_claims: int = Field(
        ..., description="Number of unsupported claims"
    )
    hallucination_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Percentage of unsupported claims"
    )
    flagged: bool = Field(
        ..., description="True if >2 unsupported claims detected"
    )
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Average confidence across all verifications"
    )
