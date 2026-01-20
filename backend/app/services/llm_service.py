"""
LLM Service with Multi-Provider Fallback (OpenAI → Gemini)

10x Principles:
1. Reliability: Automatic fallback to secondary provider
2. Observability: Comprehensive logging and metrics
3. Abstraction: Single interface for multiple LLM providers
4. Cost Optimization: Use cheaper fallback when primary fails

Architecture:
- Primary: OpenAI (gpt-4o)
- Fallback: Google Gemini (gemini-1.5-flash)
- Uses LangChain for provider abstraction
"""
import logging
import time
from typing import Dict, Any, Optional
from enum import Enum

import tiktoken
import mlflow
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from backend.app.config import get_settings

# Setup structured logging
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMService:
    """
    Multi-provider LLM service with automatic fallback.

    Flow:
    1. Try OpenAI (primary)
    2. If OpenAI fails → fallback to Gemini
    3. If both fail → raise exception

    All calls are:
    - Logged to MLflow for tracking
    - Retried with exponential backoff
    - Timed for performance monitoring
    """

    def __init__(self):
        self.settings = get_settings()

        # Initialize providers
        self.openai_available = self._init_openai()
        self.gemini_available = self._init_gemini()

        if not self.openai_available and not self.gemini_available:
            raise RuntimeError(
                "No LLM providers configured. Set OPENAI_API_KEY or GEMINI_API_KEY"
            )

        # Token encoding for cost estimation
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        # Initialize MLflow
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.experiment_name)

        logger.info(
            f"LLM Service initialized: "
            f"OpenAI={self.openai_available}, Gemini={self.gemini_available}"
        )

    def _init_openai(self) -> bool:
        """Initialize OpenAI provider if API key available."""
        try:
            if self.settings.openai_api_key is None:
                logger.warning("OPENAI_API_KEY not set - OpenAI unavailable")
                return False

            self.openai_client = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key.get_secret_value(),
                temperature=0.7,  # Default, overridden per call
                timeout=self.settings.timeout_seconds,
                max_retries=0,  # We handle retries ourselves
            )
            logger.info(f"OpenAI initialized: {self.settings.openai_model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False

    def _init_gemini(self) -> bool:
        """Initialize Gemini provider if API key available."""
        try:
            if self.settings.gemini_api_key is None:
                logger.warning("GEMINI_API_KEY not set - Gemini unavailable")
                return False

            self.gemini_client = ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=self.settings.gemini_api_key.get_secret_value(),
                temperature=0.7,  # Default, overridden per call
                timeout=self.settings.timeout_seconds,
            )
            logger.info(f"Gemini initialized: {self.settings.gemini_model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for cost tracking.
        Note: Approximation for both OpenAI and Gemini.
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using word estimate.")
            return int(len(text.split()) * 1.3)

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Call OpenAI via LangChain.

        Note: response_format is OpenAI-specific (JSON mode).
        Gemini doesn't support this directly, so we handle it differently.
        """
        # Update temperature for this call
        self.openai_client.temperature = temperature
        self.openai_client.max_tokens = max_tokens

        # Configure JSON mode if requested
        if response_format and response_format.get("type") == "json_object":
            self.openai_client.model_kwargs = {"response_format": response_format}
        else:
            self.openai_client.model_kwargs = {}

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.openai_client.invoke(messages)

        return {
            "content": response.content,
            "provider": LLMProvider.OPENAI,
            "model": self.settings.openai_model
        }

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Call Gemini via LangChain.

        For JSON mode, we append instructions to the system prompt
        since Gemini doesn't have native JSON mode.
        """
        # Update temperature for this call
        self.gemini_client.temperature = temperature
        self.gemini_client.max_output_tokens = max_tokens

        # Handle JSON mode by augmenting the prompt
        if response_format and response_format.get("type") == "json_object":
            system_prompt += "\n\nIMPORTANT: You MUST respond with valid JSON only. No additional text before or after the JSON."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.gemini_client.invoke(messages)

        return {
            "content": response.content,
            "provider": LLMProvider.GEMINI,
            "model": self.settings.gemini_model
        }

    @retry(
        retry=retry_if_exception_type((Exception,)),
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
        Generate LLM response with automatic fallback.

        Flow:
        1. Try OpenAI (if available)
        2. On failure, try Gemini (if available)
        3. If both fail, raise exception

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional format spec (e.g., {"type": "json_object"})

        Returns:
            Dict with:
            - content: Generated text
            - usage: Token counts
            - model: Model used
            - provider: Provider used
        """
        start_time = time.time()
        input_tokens = self.count_tokens(system_prompt + user_prompt)

        provider_used = None
        result_text = None
        error_chain = []

        try:
            with mlflow.start_run(nested=True, run_name="llm_generation"):
                # Try OpenAI first
                if self.openai_available:
                    try:
                        logger.debug("Attempting OpenAI...")
                        result = self._call_openai(
                            system_prompt, user_prompt, temperature,
                            max_tokens, response_format
                        )
                        result_text = result["content"]
                        provider_used = LLMProvider.OPENAI
                        mlflow.log_param("provider", "openai")
                        mlflow.log_param("model", self.settings.openai_model)
                        logger.info("✓ OpenAI succeeded")

                    except Exception as e:
                        error_msg = f"OpenAI failed: {str(e)}"
                        logger.warning(error_msg)
                        error_chain.append(error_msg)

                        # Fallback to Gemini
                        if self.gemini_available:
                            try:
                                logger.warning("→ Falling back to Gemini...")
                                result = self._call_gemini(
                                    system_prompt, user_prompt, temperature,
                                    max_tokens, response_format
                                )
                                result_text = result["content"]
                                provider_used = LLMProvider.GEMINI
                                mlflow.log_param("provider", "gemini")
                                mlflow.log_param("model", self.settings.gemini_model)
                                mlflow.log_param("fallback_used", True)
                                logger.info("✓ Gemini fallback succeeded")

                            except Exception as gemini_error:
                                error_msg = f"Gemini failed: {str(gemini_error)}"
                                logger.error(error_msg)
                                error_chain.append(error_msg)
                                raise Exception(
                                    f"All providers failed: {'; '.join(error_chain)}"
                                )
                        else:
                            raise Exception(
                                f"OpenAI failed and no fallback available: {str(e)}"
                            )

                # If OpenAI not available, use Gemini directly
                elif self.gemini_available:
                    try:
                        logger.debug("Using Gemini (OpenAI not configured)...")
                        result = self._call_gemini(
                            system_prompt, user_prompt, temperature,
                            max_tokens, response_format
                        )
                        result_text = result["content"]
                        provider_used = LLMProvider.GEMINI
                        mlflow.log_param("provider", "gemini")
                        mlflow.log_param("model", self.settings.gemini_model)
                        logger.info("✓ Gemini succeeded")

                    except Exception as e:
                        error_msg = f"Gemini failed: {str(e)}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                # Calculate metrics
                output_tokens = self.count_tokens(result_text)
                total_tokens = input_tokens + output_tokens
                duration = time.time() - start_time

                # Log metrics to MLflow
                mlflow.log_param("temperature", temperature)
                mlflow.log_param("input_tokens", input_tokens)
                mlflow.log_metric("duration_seconds", duration)
                mlflow.log_metric("output_tokens", output_tokens)
                mlflow.log_metric("total_tokens", total_tokens)

                if error_chain:
                    mlflow.log_param("errors_before_success", error_chain)

                return {
                    "content": result_text,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    },
                    "model": result["model"],
                    "provider": provider_used.value
                }

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise


def get_llm_service() -> "LLMService":
    """Factory function to get LLM service instance."""
    return LLMService()
