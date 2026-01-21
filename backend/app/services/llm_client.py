from typing import Any, Dict
import httpx

# Note: The endpoints /fact-extract, /jd-analyze, /cover-letter are placeholders. When Person A finishes their FastAPI routes, you align names.

class LLMClient:
    """
    Talks to Person A's service (or your own internal endpoints later).
    For now, it can run in stub mode if LLM_BASE_URL is not set.
    """

    def __init__(self, base_url: str | None):
        self.base_url = base_url.rstrip("/") if base_url else None

    async def extract_facts(self, cv_sections: Dict[str, str]) -> Dict[str, Any]:
        if not self.base_url:
            # stub: return minimal structure
            return {"experiences": [], "education": [], "skills": [], "raw_sections": cv_sections}

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base_url}/fact-extract", json={"sections": cv_sections})
            r.raise_for_status()
            return r.json()

    async def analyze_jd(self, job_description: str) -> Dict[str, Any]:
        if not self.base_url:
            return {"requirements": [], "keywords": [], "raw": job_description}

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base_url}/jd-analyze", json={"job_description": job_description})
            r.raise_for_status()
            return r.json()

    async def generate_cover_letter(self, facts: Dict[str, Any], jd: Dict[str, Any], tone: str) -> Dict[str, Any]:
        if not self.base_url:
            return {
                "cover_letter": "STUB COVER LETTER\n\n(LLM_BASE_URL not configured)",
                "tone": tone,
            }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{self.base_url}/cover-letter",
                json={"facts": facts, "job": jd, "tone": tone},
            )
            r.raise_for_status()
            return r.json()

def get_llm_client() -> "LLMClient":
    # Single place that decides how the client is constructed in production
    return LLMClient()
