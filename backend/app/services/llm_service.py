"""
Abstract LLM calls (OpenAI/Anthropic).
10x Principle: Reliability, Observability, and Abstraction.
"""
import logging
import time
from typing import Dict, Any, Optional, List

import tiktoken
import mlflow
from openai import OpenAI, OpenAIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from app.config import get_settings

# Setup structured logging
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.settings = get_settings()

        # Don't fail at import / construction time.
        # We'll validate only when an actual API call happens.
        self._api_key = (
            self.settings.openai_api_key.get_secret_value()
            if self.settings.openai_api_key is not None
            else None
        )

        self.client = None  # lazy init

        self.model = self.settings.openai_model
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # fallback-ish

        # MLflow init is generally fine, but keep it non-fatal for unit tests.
        try:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.settings.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow init failed (continuing): {e}")

    def _require_openai(self) -> None:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM calls")

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self._require_openai()
            self.client = OpenAI(api_key=self._api_key)
        return self.client



    def count_tokens(self, text: str) -> int:
        """
        Accurately count tokens to estimate costs and stay within limits.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Falling back to simple estimate.")
            return len(text.split()) * 1.3 # Rough estimate

    @retry(
    retry=retry_if_exception_type((Exception, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a response from the LLM with:
        1. Automatic Retries (Tenacity)
        2. Cost Tracking (MLflow)
        3. Error Handling
        """
        start_time = time.time()
        
        # Track inputs for observability
        input_tokens = self.count_tokens(system_prompt + user_prompt)
        
        try:
            # We use an MLflow run to track this specific generation event
            with mlflow.start_run(nested=True, run_name="llm_generation"):
                mlflow.log_param("model", self.model)
                mlflow.log_param("temperature", temperature)
                mlflow.log_param("input_tokens", input_tokens)
                
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,

                    timeout=self.settings.timeout_seconds,
                    response_format=response_format
                )
                
                result_text = response.choices[0].message.content
                output_tokens = response.usage.completion_tokens if response.usage else self.count_tokens(result_text)
                total_tokens = response.usage.total_tokens if response.usage else (input_tokens + output_tokens)
                
                # Log metrics
                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)
                mlflow.log_metric("output_tokens", output_tokens)
                mlflow.log_metric("total_tokens", total_tokens)
                
                # In a real 10x setup, you might log the actual text artifacts to MLflow 
                # for later inspection (be careful with PII)
                # mlflow.log_text(result_text, "output.txt")

                return {
                    "content": result_text,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    },
                    "model": self.model
                }

        except Exception as e:
            logger.error(f"LLM Generation failed after retries: {str(e)}")
            # Re-raise so the caller knows it failed, or return a fallback object
            raise e

def get_llm_service() -> "LLMService":
    return LLMService()

