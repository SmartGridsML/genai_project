import json
import logging
from typing import Any, Dict, Optional

import openai
from google import genai
from google.genai import types

from backend.app.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        settings = get_settings()

        # Gemini (required — used for structured JSON extraction and cover letter)
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._gemini = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.model = settings.gemini_model  # e.g. "gemini-2.5-flash"

        # OpenAI (optional — used as primary in generate_text() with Gemini fallback)
        self._openai: Optional[openai.AsyncOpenAI] = None
        if settings.openai_api_key:
            self._openai = openai.AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.timeout_seconds,
                max_retries=2,
            )
            logger.info(f"OpenAI client initialised ({settings.openai_model})")
        else:
            logger.info("OPENAI_API_KEY not set — generate_text() will use Gemini only")

        self._openai_model = settings.openai_model

    # ------------------------------------------------------------------
    # Structured JSON extraction (Gemini native schema enforcement)
    # ------------------------------------------------------------------

    async def _generate_json(self, *, system: str, user: str, schema: dict) -> dict:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system,
            temperature=0.2,
        )
        resp = await self._gemini.aio.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
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
        return await self._generate_json(system=system, user=f"CV TEXT:\n{cv_text}", schema=schema)

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
        return await self._generate_json(
            system="Analyze the job description and extract requirements.",
            user=f"JOB DESCRIPTION:\n{job_description}",
            schema=schema,
        )

    async def generate_cover_letter(self, *, facts: dict, jd: dict, tone: str = "professional") -> dict:
        system = "You write concise, professional cover letters grounded strictly in provided facts."
        user = (
            f"TONE: {tone}\n\n"
            f"FACTS (JSON):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"JOB ANALYSIS (JSON):\n{json.dumps(jd, ensure_ascii=False)}\n\n"
            "Write a cover letter. Do not invent details not present in FACTS."
        )
        resp = await self._gemini.aio.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.4),
        )
        return {"cover_letter": resp.text}

    # ------------------------------------------------------------------
    # Free-text generation (OpenAI primary → Gemini fallback)
    # Used by Auditor and CVEnhancer — no shared state, fully async.
    # ------------------------------------------------------------------

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text with OpenAI (if configured) → Gemini fallback.

        Set json_mode=True when the caller expects a JSON response. This enforces
        response_format={"type":"json_object"} on OpenAI and response_mime_type on
        Gemini, preventing empty or prose responses that fail json.loads().

        Returns {"content": str, "provider": "openai"|"gemini"}.
        """
        if self._openai is not None:
            try:
                extra: Dict[str, Any] = {}
                if json_mode:
                    extra["response_format"] = {"type": "json_object"}
                resp = await self._openai.chat.completions.create(
                    model=self._openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                content = resp.choices[0].message.content or ""
                if not content:
                    raise ValueError("OpenAI returned empty content")
                logger.info("generate_text: OpenAI succeeded")
                return {"content": content, "provider": "openai"}
            except Exception as e:
                logger.warning(f"generate_text: OpenAI failed ({e}), falling back to Gemini")

        # Gemini fallback — native async, no thread pool
        if json_mode:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=system_prompt,
                temperature=temperature,
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            )
        resp = await self._gemini.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        content = resp.text or ""
        if not content:
            raise RuntimeError("Gemini returned empty content")
        logger.info("generate_text: Gemini succeeded")
        return {"content": content, "provider": "gemini"}


def get_llm_client() -> LLMClient:
    return LLMClient()
