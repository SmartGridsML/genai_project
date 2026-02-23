from typing import Any, Dict
import httpx
import json
from google import genai
from google.genai import types
from backend.app.config import get_settings


# Note: The endpoints /fact-extract, /jd-analyze, /cover-letter are placeholders. When Person A finishes their FastAPI routes, you align names.

class LLMClient:
    """
    Talks to Person A's service (or your own internal endpoints later).
    For now, it can run in stub mode if LLM_BASE_URL is not set.
    """

    def __init__(self, base_url: str | None):
        self.base_url = base_url
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.model = settings.gemini_model  # e.g. "gemini-2.5-flash"
    
    async def _generate_json(self, *, system: str, user: str, schema: dict) -> dict:
        # request JSON output
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system,
            temperature=0.2,
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        # resp.text should be JSON when response_mime_type is application/json
        return json.loads(resp.text)

    async def extract_facts(self, sections: dict) -> dict:
        cv_text = "\n\n".join([f"{k.upper()}:\n{v}" for k, v in sections.items() if v])

        schema = {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "fact": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["category", "fact", "confidence"],
                    },
                }
            },
            "required": ["facts"],
        }

        system = (
            "Extract key facts from the CV text. "
            "Return only facts supported by the text. "
            "Confidence must be between 0 and 1."
        )
        user = f"CV TEXT:\n{cv_text}"

        return await self._generate_json(system=system, user=user, schema=schema)

    async def analyze_jd(self, job_description: str) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "required_skills": {"type": "array", "items": {"type": "string"}},
                "experience_level": {"type": "string"},
                "remote_policy": {"type": "string"},
            },
            "required": ["summary", "required_skills", "experience_level", "remote_policy"],
        }

        system = "Analyze the job description and extract requirements."
        user = f"JOB DESCRIPTION:\n{job_description}"

        return await self._generate_json(system=system, user=user, schema=schema)


    async def generate_cover_letter(self, *, facts: dict, jd: dict, tone: str = "professional") -> dict:
        system = "You write concise, professional cover letters grounded strictly in provided facts."
        user = (
            f"TONE: {tone}\n\n"
            f"FACTS (JSON):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"JOB ANALYSIS (JSON):\n{json.dumps(jd, ensure_ascii=False)}\n\n"
            "Write a cover letter. Do not invent details not present in FACTS."
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.4,
            ),
        )
        return {"cover_letter": resp.text}


from backend.app.config import get_settings

def get_llm_client():
    settings = get_settings()
    return LLMClient(base_url=settings.llm_base_url)

