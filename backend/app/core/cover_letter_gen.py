from __future__ import annotations

import json
from typing import Any, Dict
from pydantic import BaseModel, Field, ValidationError


COVER_LETTER_PROMPT_V1 = """You write concise, professional cover letters.

You will receive JSON with:
- FACTS: extracted CV facts (ground truth)
- JOB: job requirements/keywords (ground truth)

Write a cover letter grounded ONLY in FACTS.

Rules (critical):
- Do NOT invent employers, titles, degrees, dates, achievements, numbers, or tools not present in FACTS.
- If a requirement isn't supported by FACTS, do NOT claim it. Instead say you're eager to learn.
- Keep it 180-260 words.
- Tone: {tone}

Return STRICT JSON only (no markdown, no extra text):
{{
  "cover_letter": "string"
}}
"""


class _CoverOut(BaseModel):
    cover_letter: str = Field(..., min_length=80)


def _coerce_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError("Unexpected LLM output type")


async def generate_cover_letter(llm_service, facts: Dict[str, Any], job: Dict[str, Any], tone: str) -> str:
    payload = {"FACTS": facts, "JOB": job}
    system = COVER_LETTER_PROMPT_V1.format(tone=tone)

    resp = llm_service.generate_response(
        system_prompt=system,
        user_prompt=json.dumps(payload, ensure_ascii=False),
        temperature=0.3,
        max_tokens=700,
    )

    raw = resp.get("content", "")

    try:
        data = _coerce_json(raw)
        parsed = _CoverOut.model_validate(data)
        return parsed.cover_letter
    except (json.JSONDecodeError, ValidationError, TypeError):
        return (
            "I’m excited to apply for this role. Based on the experience described in my CV, "
            "I believe I can contribute effectively and grow into the responsibilities of the position. "
            "I would welcome the opportunity to discuss how my background aligns with your needs."
        )
