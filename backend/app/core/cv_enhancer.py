from __future__ import annotations

import json
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List

from app.core.prompts import Prompts, PromptVersion
from app.models.schemas import CVEnhancement


class CVEnhancerError(RuntimeError):
    pass


@dataclass
class CVPatch:
    section: str
    before: str
    after: str
    rationale: str
    grounded_sources: List[str]
    diff_unified: str


def _unified_diff(before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )


class CVEnhancer:
    def __init__(self, llm_service: Any):
        self.llm = llm_service

    def enhance(
        self,
        *,
        original_cv_text: str,
        fact_table: Dict[str, Any],
        jd_requirements: Dict[str, Any],
        max_suggestions: int = 8,
    ) -> List[CVPatch]:
        system_prompt = Prompts.get_cv_enhancement_system(PromptVersion.V1)

        user_prompt = json.dumps(
            {
                "max_suggestions": max_suggestions,
                "original_cv_text": original_cv_text,
                "facts": fact_table,
                "job_requirements": jd_requirements,
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            resp = self.llm.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=1400,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise CVEnhancerError(f"LLM call failed: {e}") from e

        content = resp.get("content")
        if not content:
            raise CVEnhancerError("Empty response from CV enhancer")

        # LLMService returns text; parse JSON
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except Exception as e:
            raise CVEnhancerError(f"Invalid JSON from CV enhancer: {e}") from e

        # Validate schema strictly
        try:
            parsed = CVEnhancement.model_validate(data)
        except Exception as e:
            raise CVEnhancerError(f"CVEnhancement schema validation failed: {e}") from e

        patches: List[CVPatch] = []
        for s in parsed.suggestions[:max_suggestions]:
            before = s.before.strip()
            after = s.after.strip()
            if not before or not after or before == after:
                continue

            patches.append(
                CVPatch(
                    section=s.section.strip(),
                    before=before,
                    after=after,
                    rationale=s.rationale.strip(),
                    grounded_sources=list(s.grounded_sources or []),
                    diff_unified=_unified_diff(before, after),
                )
            )

        return patches
