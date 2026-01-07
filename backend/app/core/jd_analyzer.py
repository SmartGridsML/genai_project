from __future__ import annotations

import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field, ValidationError


JD_ANALYSIS_PROMPT_V1 = """You are an expert recruiter.
Extract the job requirements from the job description.

Return STRICT JSON only (no markdown, no extra text) with this schema:
{
  "requirements": [string],
  "keywords": [string]
}

Rules:
- requirements must be concrete (skills, years, responsibilities, tools).
- keywords must be short terms (e.g., "Python", "SQL", "FastAPI").
- Do not invent anything not explicitly in the job description.
- If uncertain, return empty lists.
"""


class _JDOut(BaseModel):
    requirements: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


def _coerce_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError("Unexpected LLM output type")


async def analyze_job_description(llm_service, job_description: str) -> Dict[str, Any]:
    jd_text = job_description.strip()

    # LLMService.generate_response is sync in your code; call directly.
    # If you later make it async, change accordingly.
    resp = llm_service.generate_response(
        system_prompt=JD_ANALYSIS_PROMPT_V1,
        user_prompt=jd_text,
        temperature=0.2,
        max_tokens=700,
    )

    raw = resp.get("content", "")

    try:
        data = _coerce_json(raw)
        parsed = _JDOut.model_validate(data)
        return {"requirements": parsed.requirements, "keywords": parsed.keywords}
    except (json.JSONDecodeError, ValidationError, TypeError):
        return {"requirements": [], "keywords": []}
