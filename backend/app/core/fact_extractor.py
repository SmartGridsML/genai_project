"""CV → JSON facts (LLM)
Extracts structured facts from CV text using LLM with validation.
"""

import json
import logging

import mlflow
from pydantic import ValidationError

from backend.app.core.prompts import PromptVersion, Prompts
from backend.app.services.llm_service import LLMService
from backend.app.models.schemas import ExtractedFacts

logger = logging.getLogger(__name__)


class FactExtractionError(Exception):
    """Raised when fact extraction fails."""
    pass


class FactExtractor:
    """Extracts structured facts from CV text using LLM."""

    def __init__(self, llm_service: LLMService = None):
        """Initialize with optional LLM service for dependency injection."""
        self.llm = llm_service or LLMService()
        self.version = PromptVersion.V1

    def extract(
        self,
        cv_text: str,
        request_id: str = None
    ) -> ExtractedFacts:
        """
        Extract structured facts from CV text.

        Args:
            cv_text: The CV text to extract facts from
            request_id: Optional request ID for tracking

        Returns:
            ExtractedFacts: Validated structured facts

        Raises:
            FactExtractionError: If extraction or validation fails
        """
        if not cv_text or not cv_text.strip():
            raise FactExtractionError("CV text cannot be empty")

        try:
            with mlflow.start_run(run_name="fact_extraction", nested=True):
                # Log extraction parameters
                mlflow.log_param("prompt_version", self.version.value)
                mlflow.log_param("cv_length", len(cv_text))
                if request_id:
                    mlflow.log_param("request_id", request_id)

                # Get system prompt
                system_prompt = Prompts.get_fact_extraction_system(
                    self.version
                )

                # Call LLM with JSON mode
                cv_len = len(cv_text)
                logger.info(f"Extracting facts from CV (length: {cv_len})")
                response = self.llm.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=cv_text,
                    temperature=0.3,  # Lower temp for consistency
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )

                # Parse JSON response
                raw_content = response["content"]
                
                # Clean up potential Markdown formatting (common with Gemini)
                if raw_content.strip().startswith("```"):
                    # Remove opening ```json or ```
                    raw_content = raw_content.split("\n", 1)[1]
                    # Remove closing ```
                    if raw_content.strip().endswith("```"):
                        raw_content = raw_content.rsplit("```", 1)[0]
                
                try:
                    parsed_json = json.loads(raw_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.error(f"Raw response: {raw_content[:500]}")
                    mlflow.log_text(raw_content, "failed_response.txt")
                    raise FactExtractionError(f"LLM returned invalid JSON: {str(e)}")

                # Validate against Pydantic schema
                try:
                    extracted_facts = ExtractedFacts.model_validate(
                        parsed_json
                    )
                except ValidationError as e:
                    logger.error(f"Schema validation failed: {e}")
                    logger.error(f"Parsed JSON: {parsed_json}")
                    invalid_json = json.dumps(parsed_json, indent=2)
                    mlflow.log_text(invalid_json, "invalid_schema.json")
                    msg = f"Response doesn't match schema: {str(e)}"
                    raise FactExtractionError(msg)

                # Log extraction metrics
                fact_count = len(extracted_facts.facts)
                mlflow.log_metric("facts_extracted", fact_count)

                if fact_count > 0:
                    confidences = [
                        f.confidence for f in extracted_facts.facts
                    ]
                    avg_confidence = sum(confidences) / fact_count
                    mlflow.log_metric("avg_confidence", avg_confidence)

                    # Log category distribution
                    categories = {}
                    for fact in extracted_facts.facts:
                        cat = fact.category
                        categories[cat] = categories.get(cat, 0) + 1
                    mlflow.log_dict(
                        categories,
                        "category_distribution.json"
                    )

                # Log successful extraction
                mlflow.log_text(raw_content, "extracted_facts.json")
                logger.info(f"Successfully extracted {fact_count} facts")

                return extracted_facts

        except FactExtractionError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Catch any unexpected errors
            logger.error(
                f"Unexpected error during fact extraction: {e}",
                exc_info=True
            )
            raise FactExtractionError(
                f"Fact extraction failed: {str(e)}"
            )


# Singleton instance for easy import
fact_extractor = FactExtractor()
