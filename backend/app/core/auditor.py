"""
Auditor: Anti-Hallucination System for Cover Letter Generation

This is the SECRET WEAPON - a rigorous verification system that prevents
the LLM from inventing facts not present in the CV.

10x Principles Applied:
1. Defense in Depth: Multiple verification layers
2. Observability: Comprehensive metrics and logging
3. Fail-Safe: Flag suspicious content proactively
4. Auditability: Clear evidence trail for every claim

Author: Person A
"""

import json
import logging
from typing import List

import mlflow
from pydantic import ValidationError

from app.core.prompts import PromptVersion, Prompts
from app.services.llm_service import LLMService
from app.models.schemas import (
    ExtractedFacts,
    ClaimVerification,
    AuditReport
)

logger = logging.getLogger(__name__)


class AuditorError(Exception):
    """Raised when auditing fails."""
    pass


class Auditor:
    """
    Audits generated cover letters to prevent hallucinations.

    The auditor works in two phases:
    1. Claim Extraction: Identify all factual claims in the cover letter
    2. Claim Verification: Verify each claim against the fact table

    This is a 10x engineer's approach to the hallucination problem:
    - Don't just hope the LLM is accurate
    - Actively verify and flag issues
    - Provide actionable feedback
    """

    # Threshold for flagging a letter as problematic
    UNSUPPORTED_CLAIM_THRESHOLD = 2

    def __init__(self, llm_service: LLMService = None):
        """Initialize with optional LLM service for dependency injection."""
        self.llm = llm_service or LLMService()
        self.version = PromptVersion.V1

    def audit(
        self,
        cover_letter: str,
        fact_table: ExtractedFacts,
        request_id: str = None
    ) -> AuditReport:
        """
        Audit a cover letter against the fact table.

        Args:
            cover_letter: The generated cover letter to audit
            fact_table: The extracted facts from the CV
            request_id: Optional request ID for tracking

        Returns:
            AuditReport: Complete audit results with flagging

        Raises:
            AuditorError: If auditing fails
        """
        if not cover_letter or not cover_letter.strip():
            raise AuditorError("Cover letter cannot be empty")

        if not fact_table or not fact_table.facts:
            raise AuditorError("Fact table cannot be empty")

        try:
            with mlflow.start_run(run_name="audit", nested=True):
                # Log audit parameters
                mlflow.log_param("prompt_version", self.version.value)
                mlflow.log_param("cover_letter_length", len(cover_letter))
                mlflow.log_param("fact_count", len(fact_table.facts))
                if request_id:
                    mlflow.log_param("request_id", request_id)

                # Phase 1: Extract claims from cover letter
                logger.info("Phase 1: Extracting claims from cover letter")
                claims = self._extract_claims(cover_letter)
                mlflow.log_metric("claims_extracted", len(claims))

                # Phase 2: Verify each claim
                logger.info(
                    f"Phase 2: Verifying {len(claims)} claims "
                    f"against {len(fact_table.facts)} facts"
                )
                verifications = self._verify_claims(claims, fact_table)

                # Phase 3: Generate audit report
                report = self._generate_report(verifications)

                # Log audit metrics
                self._log_audit_metrics(report)

                logger.info(
                    f"Audit complete: {report.supported_claims}/"
                    f"{report.total_claims} claims supported, "
                    f"hallucination rate: {report.hallucination_rate:.2%}"
                )

                if report.flagged:
                    logger.warning(
                        f"FLAGGED: Cover letter contains "
                        f"{report.unsupported_claims} unsupported claims "
                        f"(threshold: {self.UNSUPPORTED_CLAIM_THRESHOLD})"
                    )

                return report

        except AuditorError:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during audit: {e}",
                exc_info=True
            )
            raise AuditorError(f"Audit failed: {str(e)}")

    def _extract_claims(self, cover_letter: str) -> List[str]:
        """
        Extract factual claims from the cover letter.

        Returns:
            List of claim strings
        """
        system_prompt = Prompts.get_claim_extraction_system(self.version)

        try:
            response = self.llm.generate_response(
                system_prompt=system_prompt,
                user_prompt=cover_letter,
                temperature=0.2,  # Low temp for consistency
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            raw_content = response["content"]
            parsed_json = json.loads(raw_content)

            # Validate response structure
            if "claims" not in parsed_json:
                raise AuditorError(
                    "LLM response missing 'claims' field"
                )

            claims = parsed_json["claims"]
            if not isinstance(claims, list):
                raise AuditorError(
                    "'claims' field must be a list"
                )

            # Filter out empty claims
            claims = [c.strip() for c in claims if c and c.strip()]

            logger.info(f"Extracted {len(claims)} claims")
            mlflow.log_text(
                json.dumps(claims, indent=2),
                "extracted_claims.json"
            )

            return claims

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse claim extraction response: {e}")
            raise AuditorError(
                f"LLM returned invalid JSON: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            raise AuditorError(
                f"Failed to extract claims: {str(e)}"
            )

    def _verify_claims(
        self,
        claims: List[str],
        fact_table: ExtractedFacts
    ) -> List[ClaimVerification]:
        """
        Verify each claim against the fact table.

        Returns:
            List of ClaimVerification objects
        """
        verifications = []
        system_prompt = Prompts.get_claim_verification_system(self.version)

        # Convert fact table to JSON for LLM
        facts_json = json.dumps(
            [fact.model_dump() for fact in fact_table.facts],
            indent=2
        )

        for i, claim in enumerate(claims):
            logger.debug(f"Verifying claim {i+1}/{len(claims)}: {claim}")

            # Construct verification prompt
            user_prompt = f"""
CLAIM: {claim}

FACTS:
{facts_json}
"""

            try:
                response = self.llm.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,  # Very low temp for strict verification
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                raw_content = response["content"]
                parsed_json = json.loads(raw_content)

                # Create verification object
                verification = ClaimVerification(
                    claim=claim,
                    supported=parsed_json.get("supported", False),
                    source=parsed_json.get("source", "UNSUPPORTED"),
                    confidence=parsed_json.get("confidence", 0.0),
                    reasoning=parsed_json.get("reasoning", "")
                )

                verifications.append(verification)

                logger.debug(
                    f"Claim '{claim}': "
                    f"supported={verification.supported}, "
                    f"confidence={verification.confidence:.2f}"
                )

            except (json.JSONDecodeError, ValidationError) as e:
                # If verification fails, mark as unsupported
                logger.error(
                    f"Failed to verify claim '{claim}': {e}"
                )
                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        supported=False,
                        source="VERIFICATION_FAILED",
                        confidence=0.0,
                        reasoning=f"Verification error: {str(e)}"
                    )
                )

        # Log verification results
        mlflow.log_text(
            json.dumps(
                [v.model_dump() for v in verifications],
                indent=2
            ),
            "verifications.json"
        )

        return verifications

    def _generate_report(
        self,
        verifications: List[ClaimVerification]
    ) -> AuditReport:
        """
        Generate the final audit report with metrics.

        Returns:
            AuditReport object
        """
        total_claims = len(verifications)
        supported_claims = sum(1 for v in verifications if v.supported)
        unsupported_claims = total_claims - supported_claims

        # Calculate hallucination rate
        hallucination_rate = (
            unsupported_claims / total_claims if total_claims > 0 else 0.0
        )

        # Calculate overall confidence
        if total_claims > 0:
            overall_confidence = sum(
                v.confidence for v in verifications
            ) / total_claims
        else:
            overall_confidence = 0.0

        # Flag if too many unsupported claims
        flagged = unsupported_claims > self.UNSUPPORTED_CLAIM_THRESHOLD

        report = AuditReport(
            verifications=verifications,
            total_claims=total_claims,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            hallucination_rate=hallucination_rate,
            flagged=flagged,
            overall_confidence=overall_confidence
        )

        return report

    def _log_audit_metrics(self, report: AuditReport) -> None:
        """Log comprehensive metrics to MLflow."""
        mlflow.log_metric("total_claims", report.total_claims)
        mlflow.log_metric("supported_claims", report.supported_claims)
        mlflow.log_metric("unsupported_claims", report.unsupported_claims)
        mlflow.log_metric("hallucination_rate", report.hallucination_rate)
        mlflow.log_metric("overall_confidence", report.overall_confidence)
        mlflow.log_metric("flagged", int(report.flagged))

        # Log distribution of confidence scores
        if report.verifications:
            confidences = [v.confidence for v in report.verifications]
            mlflow.log_metric("min_confidence", min(confidences))
            mlflow.log_metric("max_confidence", max(confidences))

            # Log unsupported claim types for analysis
            unsupported_claims = [
                v.claim for v in report.verifications if not v.supported
            ]
            if unsupported_claims:
                mlflow.log_text(
                    json.dumps(unsupported_claims, indent=2),
                    "unsupported_claims.json"
                )


# Singleton instance
#auditor = Auditor()
