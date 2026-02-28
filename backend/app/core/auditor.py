"""
Auditor: Anti-Hallucination System for Cover Letter Generation
"""

import json
import logging
from contextlib import nullcontext
from typing import List, TYPE_CHECKING

import mlflow
from pydantic import ValidationError

from backend.app.config import get_settings
from backend.app.core.prompts import PromptVersion, Prompts
from backend.app.models.schemas import (
    ExtractedFacts,
    ClaimVerification,
    AuditReport,
)

if TYPE_CHECKING:
    from backend.app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AuditorError(Exception):
    """Raised when auditing fails."""
    pass


class Auditor:
    """
    Audits generated cover letters to prevent hallucinations.

    Phase 1: Extract all factual claims from the cover letter.
    Phase 2: Verify each claim against the extracted fact table.
    """

    UNSUPPORTED_CLAIM_THRESHOLD = 2

    def __init__(self, llm_client: "LLMClient"):
        self.llm = llm_client
        self.version = PromptVersion.V1
        self._mlflow = get_settings().mlflow_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def audit(
        self,
        cover_letter: str,
        fact_table: ExtractedFacts,
        request_id: str = None,
    ) -> AuditReport:
        if not cover_letter or not cover_letter.strip():
            raise AuditorError("Cover letter cannot be empty")
        if not fact_table or not fact_table.facts:
            raise AuditorError("Fact table cannot be empty")

        try:
            if self._mlflow:
                mlflow.set_experiment("audit")
            ctx = mlflow.start_run(run_name="audit") if self._mlflow else nullcontext()
            with ctx:
                if self._mlflow:
                    mlflow.log_param("prompt_version", self.version.value)
                    mlflow.log_param("cover_letter_length", len(cover_letter))
                    mlflow.log_param("fact_count", len(fact_table.facts))
                    if request_id:
                        mlflow.log_param("request_id", request_id)

                logger.info("Phase 1: Extracting claims from cover letter")
                claims = await self._extract_claims(cover_letter)
                if self._mlflow:
                    mlflow.log_metric("claims_extracted", len(claims))

                logger.info(
                    f"Phase 2: Verifying {len(claims)} claims "
                    f"against {len(fact_table.facts)} facts"
                )
                verifications = await self._verify_claims_batch(claims, fact_table)

                report = self._generate_report(verifications)
                self._log_audit_metrics(report)

                logger.info(
                    f"Audit complete: {report.supported_claims}/"
                    f"{report.total_claims} claims supported, "
                    f"hallucination rate: {report.hallucination_rate:.2%}"
                )
                if report.flagged:
                    logger.warning(
                        f"FLAGGED: {report.unsupported_claims} unsupported claims "
                        f"(threshold: {self.UNSUPPORTED_CLAIM_THRESHOLD})"
                    )
                return report

        except AuditorError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during audit: {e}", exc_info=True)
            raise AuditorError(f"Audit failed: {str(e)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_claims(self, cover_letter: str) -> List[str]:
        system_prompt = Prompts.get_claim_extraction_system(self.version)
        try:
            response = await self.llm.generate_text(
                system_prompt=system_prompt,
                user_prompt=cover_letter,
                temperature=0.2,
                max_tokens=1000,
                json_mode=True,
            )
            raw_content = response["content"]

            if raw_content.strip().startswith("```"):
                raw_content = raw_content.split("\n", 1)[1]
                if raw_content.strip().endswith("```"):
                    raw_content = raw_content.rsplit("```", 1)[0]

            parsed_json = json.loads(raw_content)

            if "claims" not in parsed_json:
                raise AuditorError("LLM response missing 'claims' field")
            claims = parsed_json["claims"]
            if not isinstance(claims, list):
                raise AuditorError("'claims' field must be a list")

            claims = [c.strip() for c in claims if c and c.strip()]
            logger.info(f"Extracted {len(claims)} claims")

            if len(claims) == 0:
                logger.warning(f"No claims extracted! Raw response: {raw_content[:500]}")
                if self._mlflow:
                    mlflow.log_text(raw_content, "empty_claims_response.txt")
            if self._mlflow:
                mlflow.log_text(json.dumps(claims, indent=2), "extracted_claims.json")

            return claims

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse claim extraction response: {e}")
            raise AuditorError(f"LLM returned invalid JSON: {str(e)}")
        except AuditorError:
            raise
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            raise AuditorError(f"Failed to extract claims: {str(e)}")

    async def _verify_claims_batch(
        self,
        claims: List[str],
        fact_table: ExtractedFacts,
    ) -> List[ClaimVerification]:
        """Verify all claims in a single LLM call. Falls back to sequential on parse failure."""
        if not claims:
            return []

        system_prompt = Prompts.get_batch_claim_verification_system(self.version)
        user_prompt = json.dumps(
            {
                "claims": claims,
                "facts": [f.model_dump() for f in fact_table.facts],
            },
            ensure_ascii=False,
        )

        try:
            response = await self.llm.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=4000,
                json_mode=True,
            )
            raw = response["content"]
            if raw.strip().startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.strip().endswith("```"):
                    raw = raw.rsplit("```", 1)[0]

            data = json.loads(raw)
            verifications_raw = data.get("verifications", [])
            if not isinstance(verifications_raw, list) or len(verifications_raw) != len(claims):
                raise ValueError(
                    f"Expected {len(claims)} verifications, got {len(verifications_raw)}"
                )

            verifications = [
                ClaimVerification(
                    claim=v.get("claim", claims[i]),
                    supported=v.get("supported", False),
                    source=v.get("source", "UNSUPPORTED"),
                    confidence=float(v.get("confidence", 0.0)),
                    reasoning=v.get("reasoning", ""),
                )
                for i, v in enumerate(verifications_raw)
            ]

        except Exception as e:
            logger.warning(f"Batch verification failed ({e}), falling back to sequential")
            verifications = await self._verify_claims_sequential(claims, fact_table)

        if self._mlflow:
            mlflow.log_text(
                json.dumps([v.model_dump() for v in verifications], indent=2),
                "verifications.json",
            )
        return verifications

    async def _verify_claims_sequential(
        self,
        claims: List[str],
        fact_table: ExtractedFacts,
    ) -> List[ClaimVerification]:
        """Sequential per-claim fallback."""
        verifications = []
        system_prompt = Prompts.get_claim_verification_system(self.version)
        facts_json = json.dumps([f.model_dump() for f in fact_table.facts], indent=2)

        for i, claim in enumerate(claims):
            logger.debug(f"Verifying claim {i+1}/{len(claims)}: {claim}")
            user_prompt = f"CLAIM: {claim}\n\nFACTS:\n{facts_json}"
            try:
                response = await self.llm.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,
                    max_tokens=500,
                    json_mode=True,
                )
                raw_content = response["content"]
                if raw_content.strip().startswith("```"):
                    raw_content = raw_content.split("\n", 1)[1]
                    if raw_content.strip().endswith("```"):
                        raw_content = raw_content.rsplit("```", 1)[0]

                parsed_json = json.loads(raw_content)
                verifications.append(ClaimVerification(
                    claim=claim,
                    supported=parsed_json.get("supported", False),
                    source=parsed_json.get("source", "UNSUPPORTED"),
                    confidence=parsed_json.get("confidence", 0.0),
                    reasoning=parsed_json.get("reasoning", ""),
                ))
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                logger.error(f"Failed to verify claim '{claim}': {e}")
                verifications.append(ClaimVerification(
                    claim=claim,
                    supported=False,
                    source="VERIFICATION_FAILED",
                    confidence=0.0,
                    reasoning=f"Verification error: {str(e)}",
                ))
        return verifications

    def _generate_report(self, verifications: List[ClaimVerification]) -> AuditReport:
        total_claims = len(verifications)
        supported_claims = sum(1 for v in verifications if v.supported)
        unsupported_claims = total_claims - supported_claims
        hallucination_rate = unsupported_claims / total_claims if total_claims > 0 else 0.0
        overall_confidence = (
            sum(v.confidence for v in verifications) / total_claims
            if total_claims > 0 else 0.0
        )
        flagged = unsupported_claims > self.UNSUPPORTED_CLAIM_THRESHOLD
        return AuditReport(
            verifications=verifications,
            total_claims=total_claims,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            hallucination_rate=hallucination_rate,
            flagged=flagged,
            overall_confidence=overall_confidence,
        )

    def _log_audit_metrics(self, report: AuditReport) -> None:
        if not self._mlflow:
            return
        mlflow.log_metric("total_claims", report.total_claims)
        mlflow.log_metric("supported_claims", report.supported_claims)
        mlflow.log_metric("unsupported_claims", report.unsupported_claims)
        mlflow.log_metric("hallucination_rate", report.hallucination_rate)
        mlflow.log_metric("overall_confidence", report.overall_confidence)
        mlflow.log_metric("flagged", int(report.flagged))
        if report.verifications:
            confidences = [v.confidence for v in report.verifications]
            mlflow.log_metric("min_confidence", min(confidences))
            mlflow.log_metric("max_confidence", max(confidences))
            unsupported = [v.claim for v in report.verifications if not v.supported]
            if unsupported:
                mlflow.log_text(json.dumps(unsupported, indent=2), "unsupported_claims.json")
