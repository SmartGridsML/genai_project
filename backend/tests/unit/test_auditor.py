"""
Unit tests for Auditor

Tests the anti-hallucination system that verifies cover letters
against the fact table.

10x Testing Principles:
1. Test all failure modes
2. Test boundary conditions
3. Test integration points
4. Verify observability (MLflow logging)
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from app.core.auditor import Auditor, AuditorError
from app.models.schemas import (
    ExtractedFacts,
    KeyFact,
    ClaimVerification,
    AuditReport
)


# Test data fixtures
@pytest.fixture
def sample_fact_table():
    """Sample fact table extracted from a CV."""
    return ExtractedFacts(
        facts=[
            KeyFact(
                category="experience",
                fact="Senior Software Engineer at Google, 2020-2023",
                confidence=0.95
            ),
            KeyFact(
                category="experience",
                fact="Led a team of 5 engineers",
                confidence=0.90
            ),
            KeyFact(
                category="skills",
                fact="Expert in Python, FastAPI, and Docker",
                confidence=0.92
            ),
            KeyFact(
                category="education",
                fact="BS Computer Science, MIT, 2018",
                confidence=0.98
            )
        ]
    )


@pytest.fixture
def truthful_cover_letter():
    """Cover letter with only supported claims."""
    return """
    I am excited to apply for this position. During my time as a
    Senior Software Engineer at Google from 2020-2023, I led a team
    of 5 engineers and gained expertise in Python, FastAPI, and Docker.
    I hold a BS in Computer Science from MIT (2018).
    """


@pytest.fixture
def hallucinated_cover_letter():
    """Cover letter with unsupported claims (hallucinations)."""
    return """
    I am excited to apply for this position. During my time as a
    Principal Engineer at Amazon from 2015-2023, I led a team of
    20 developers and gained expertise in Java, Spring Boot, and
    Kubernetes. I hold a PhD in Computer Science from Stanford.
    """


@pytest.fixture
def mock_llm_service():
    """Mock LLMService for dependency injection."""
    return MagicMock()


@pytest.fixture
def mock_claim_extraction_response():
    """Mock LLM response for claim extraction."""
    return {
        "content": json.dumps({
            "claims": [
                "Senior Software Engineer at Google",
                "Worked from 2020-2023",
                "Led a team of 5 engineers",
                "Expert in Python, FastAPI, and Docker",
                "BS Computer Science from MIT",
                "Graduated in 2018"
            ]
        }),
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "model": "gpt-4"
    }


@pytest.fixture
def mock_verification_response_supported():
    """Mock LLM response for supported claim verification."""
    return {
        "content": json.dumps({
            "supported": True,
            "source": "CV fact: Senior Software Engineer at Google, 2020-2023",
            "confidence": 1.0,
            "reasoning": "Exact match with CV fact"
        }),
        "usage": {"input_tokens": 50, "output_tokens": 30},
        "model": "gpt-4"
    }


@pytest.fixture
def mock_verification_response_unsupported():
    """Mock LLM response for unsupported claim verification."""
    return {
        "content": json.dumps({
            "supported": False,
            "source": "UNSUPPORTED",
            "confidence": 0.1,
            "reasoning": "No matching fact in CV"
        }),
        "usage": {"input_tokens": 50, "output_tokens": 30},
        "model": "gpt-4"
    }


class TestAuditorInit:
    """Test initialization and dependency injection."""

    def test_init_without_llm_service(self):
        """Should create its own LLMService if none provided."""
        auditor = Auditor()
        assert auditor.llm is not None
        assert hasattr(auditor, 'version')
        assert auditor.UNSUPPORTED_CLAIM_THRESHOLD == 2

    def test_init_with_llm_service(self, mock_llm_service):
        """Should use injected LLMService for testing."""
        auditor = Auditor(llm_service=mock_llm_service)
        assert auditor.llm == mock_llm_service


class TestAuditorHappyPath:
    """Test successful auditing scenarios."""

    @patch('app.core.auditor.mlflow')
    def test_audit_truthful_cover_letter(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table,
        mock_claim_extraction_response,
        mock_verification_response_supported
    ):
        """
        Happy path: Truthful cover letter with all supported claims.

        This tests the complete flow:
        1. Extract claims from cover letter
        2. Verify each claim against fact table
        3. Generate audit report
        4. All claims supported -> not flagged
        """
        # Arrange
        mock_llm_service.generate_response.side_effect = [
            mock_claim_extraction_response,  # Claim extraction
            mock_verification_response_supported,  # Verification 1
            mock_verification_response_supported,  # Verification 2
            mock_verification_response_supported,  # Verification 3
            mock_verification_response_supported,  # Verification 4
            mock_verification_response_supported,  # Verification 5
            mock_verification_response_supported,  # Verification 6
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(
            truthful_cover_letter,
            sample_fact_table,
            request_id="test-123"
        )

        # Assert
        assert isinstance(report, AuditReport)
        assert report.total_claims == 6
        assert report.supported_claims == 6
        assert report.unsupported_claims == 0
        assert report.hallucination_rate == 0.0
        assert report.flagged is False
        assert report.overall_confidence > 0.9

        # Verify LLM was called 7 times (1 extraction + 6 verifications)
        assert mock_llm_service.generate_response.call_count == 7

        # Verify MLflow logging
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()

    @patch('app.core.auditor.mlflow')
    def test_audit_hallucinated_cover_letter(
        self,
        mock_mlflow,
        mock_llm_service,
        hallucinated_cover_letter,
        sample_fact_table
    ):
        """
        Test auditing a cover letter with hallucinations.

        Expected: Multiple unsupported claims -> flagged
        """
        # Arrange - Extract claims
        claim_response = {
            "content": json.dumps({
                "claims": [
                    "Principal Engineer at Amazon",
                    "Worked from 2015-2023",
                    "Led a team of 20 developers",
                    "Expert in Java and Spring Boot"
                ]
            }),
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "model": "gpt-4"
        }

        # All verifications return unsupported
        unsupported = {
            "content": json.dumps({
                "supported": False,
                "source": "UNSUPPORTED",
                "confidence": 0.2,
                "reasoning": "No matching fact in CV"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.side_effect = [
            claim_response,
            unsupported,
            unsupported,
            unsupported,
            unsupported
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(
            hallucinated_cover_letter,
            sample_fact_table
        )

        # Assert
        assert isinstance(report, AuditReport)
        assert report.total_claims == 4
        assert report.unsupported_claims == 4
        assert report.hallucination_rate == 1.0
        assert report.flagged is True  # >2 unsupported claims
        assert report.overall_confidence < 0.3

    @patch('app.core.auditor.mlflow')
    def test_audit_mixed_claims(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_fact_table
    ):
        """Test cover letter with mix of supported and unsupported claims."""
        # Arrange
        mixed_letter = "I worked at Google and Amazon."

        claim_response = {
            "content": json.dumps({
                "claims": ["Worked at Google", "Worked at Amazon"]
            }),
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": "gpt-4"
        }

        supported = {
            "content": json.dumps({
                "supported": True,
                "source": "CV fact: Google",
                "confidence": 0.9,
                "reasoning": "Found in CV"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        unsupported = {
            "content": json.dumps({
                "supported": False,
                "source": "UNSUPPORTED",
                "confidence": 0.3,
                "reasoning": "Not in CV"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.side_effect = [
            claim_response,
            supported,
            unsupported
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(mixed_letter, sample_fact_table)

        # Assert
        assert report.total_claims == 2
        assert report.supported_claims == 1
        assert report.unsupported_claims == 1
        assert report.hallucination_rate == 0.5
        assert report.flagged is False  # Only 1 unsupported (<=2)


class TestAuditorErrorHandling:
    """Test error handling for various failure modes."""

    def test_audit_empty_cover_letter(
        self,
        mock_llm_service,
        sample_fact_table
    ):
        """Should raise AuditorError for empty cover letter."""
        auditor = Auditor(llm_service=mock_llm_service)

        with pytest.raises(AuditorError) as exc_info:
            auditor.audit("", sample_fact_table)

        assert "cannot be empty" in str(exc_info.value)
        mock_llm_service.generate_response.assert_not_called()

    def test_audit_whitespace_only_cover_letter(
        self,
        mock_llm_service,
        sample_fact_table
    ):
        """Should raise AuditorError for whitespace-only cover letter."""
        auditor = Auditor(llm_service=mock_llm_service)

        with pytest.raises(AuditorError) as exc_info:
            auditor.audit("   \n  \t  ", sample_fact_table)

        assert "cannot be empty" in str(exc_info.value)

    def test_audit_empty_fact_table(
        self,
        mock_llm_service,
        truthful_cover_letter
    ):
        """Should raise AuditorError for empty fact table."""
        auditor = Auditor(llm_service=mock_llm_service)
        empty_facts = ExtractedFacts(facts=[])

        with pytest.raises(AuditorError) as exc_info:
            auditor.audit(truthful_cover_letter, empty_facts)

        assert "Fact table cannot be empty" in str(exc_info.value)

    @patch('app.core.auditor.mlflow')
    def test_audit_invalid_claim_extraction_json(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table
    ):
        """Should handle invalid JSON from claim extraction."""
        # Arrange
        invalid_response = {
            "content": "This is not valid JSON {broken",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = invalid_response
        auditor = Auditor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(AuditorError) as exc_info:
            auditor.audit(truthful_cover_letter, sample_fact_table)

        assert "invalid JSON" in str(exc_info.value)

    @patch('app.core.auditor.mlflow')
    def test_audit_missing_claims_field(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table
    ):
        """Should handle response missing 'claims' field."""
        # Arrange
        missing_field_response = {
            "content": json.dumps({"wrong_field": ["claim1"]}),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = (
            missing_field_response
        )
        auditor = Auditor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(AuditorError) as exc_info:
            auditor.audit(truthful_cover_letter, sample_fact_table)

        assert "missing 'claims' field" in str(exc_info.value)

    @patch('app.core.auditor.mlflow')
    def test_audit_claims_not_list(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table
    ):
        """Should handle 'claims' field that's not a list."""
        # Arrange
        wrong_type_response = {
            "content": json.dumps({"claims": "not a list"}),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = (
            wrong_type_response
        )
        auditor = Auditor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(AuditorError) as exc_info:
            auditor.audit(truthful_cover_letter, sample_fact_table)

        assert "must be a list" in str(exc_info.value)

    @patch('app.core.auditor.mlflow')
    def test_audit_verification_failure_fallback(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table
    ):
        """
        Should handle verification failures gracefully.

        If a single claim verification fails, mark it as unsupported
        and continue with other claims.
        """
        # Arrange
        claim_response = {
            "content": json.dumps({
                "claims": ["Claim 1", "Claim 2"]
            }),
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": "gpt-4"
        }

        valid_verification = {
            "content": json.dumps({
                "supported": True,
                "source": "CV fact",
                "confidence": 0.9,
                "reasoning": "Found"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        # Second verification returns invalid JSON
        invalid_verification = {
            "content": "Invalid JSON {",
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.side_effect = [
            claim_response,
            valid_verification,
            invalid_verification
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(truthful_cover_letter, sample_fact_table)

        # Assert
        assert report.total_claims == 2
        assert report.supported_claims == 1
        assert report.unsupported_claims == 1
        # Second claim should be marked unsupported due to failure
        assert report.verifications[1].supported is False
        assert "VERIFICATION_FAILED" in report.verifications[1].source


class TestAuditorBoundaryConditions:
    """Test boundary conditions and edge cases."""

    @patch('app.core.auditor.mlflow')
    def test_audit_exactly_threshold_unsupported_claims(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_fact_table
    ):
        """
        Test the exact threshold (2 unsupported claims).

        Should NOT be flagged (only >2 is flagged).
        """
        # Arrange
        cover_letter = "Test letter"

        claim_response = {
            "content": json.dumps({
                "claims": ["Claim 1", "Claim 2", "Claim 3"]
            }),
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": "gpt-4"
        }

        supported = {
            "content": json.dumps({
                "supported": True,
                "source": "CV",
                "confidence": 0.9,
                "reasoning": "OK"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        unsupported = {
            "content": json.dumps({
                "supported": False,
                "source": "UNSUPPORTED",
                "confidence": 0.2,
                "reasoning": "Not found"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.side_effect = [
            claim_response,
            supported,  # 1 supported
            unsupported,  # 1 unsupported
            unsupported  # 2 unsupported (exactly at threshold)
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(cover_letter, sample_fact_table)

        # Assert
        assert report.unsupported_claims == 2
        assert report.flagged is False  # Exactly 2, not >2

    @patch('app.core.auditor.mlflow')
    def test_audit_one_over_threshold(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_fact_table
    ):
        """Test with 3 unsupported claims (one over threshold)."""
        # Arrange
        cover_letter = "Test letter"

        claim_response = {
            "content": json.dumps({
                "claims": ["C1", "C2", "C3", "C4"]
            }),
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": "gpt-4"
        }

        supported = {
            "content": json.dumps({
                "supported": True,
                "source": "CV",
                "confidence": 0.9,
                "reasoning": "OK"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        unsupported = {
            "content": json.dumps({
                "supported": False,
                "source": "UNSUPPORTED",
                "confidence": 0.2,
                "reasoning": "Not found"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.side_effect = [
            claim_response,
            supported,
            unsupported,
            unsupported,
            unsupported  # 3 unsupported (>2)
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(cover_letter, sample_fact_table)

        # Assert
        assert report.unsupported_claims == 3
        assert report.flagged is True

    @patch('app.core.auditor.mlflow')
    def test_audit_no_claims_extracted(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_fact_table
    ):
        """Test cover letter with no factual claims."""
        # Arrange
        cover_letter = "I am very interested in this role."

        claim_response = {
            "content": json.dumps({
                "claims": []
            }),
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.return_value = claim_response
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        report = auditor.audit(cover_letter, sample_fact_table)

        # Assert
        assert report.total_claims == 0
        assert report.hallucination_rate == 0.0
        assert report.flagged is False


class TestAuditorMLflowLogging:
    """Test MLflow logging functionality."""

    @patch('app.core.auditor.mlflow')
    def test_logs_audit_parameters(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table,
        mock_claim_extraction_response,
        mock_verification_response_supported
    ):
        """Should log audit parameters to MLflow."""
        # Arrange
        mock_llm_service.generate_response.side_effect = [
            mock_claim_extraction_response,
            mock_verification_response_supported,
            mock_verification_response_supported,
            mock_verification_response_supported,
            mock_verification_response_supported,
            mock_verification_response_supported,
            mock_verification_response_supported
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        auditor.audit(
            truthful_cover_letter,
            sample_fact_table,
            request_id="test-123"
        )

        # Assert
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()

    @patch('app.core.auditor.mlflow')
    def test_logs_comprehensive_metrics(
        self,
        mock_mlflow,
        mock_llm_service,
        truthful_cover_letter,
        sample_fact_table,
        mock_claim_extraction_response,
        mock_verification_response_supported
    ):
        """Should log comprehensive audit metrics."""
        # Arrange
        mock_llm_service.generate_response.side_effect = [
            mock_claim_extraction_response,
        ] + [mock_verification_response_supported] * 6

        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        auditor.audit(truthful_cover_letter, sample_fact_table)

        # Assert - Check that key metrics were logged
        # In a real test, you'd verify specific metric names
        assert mock_mlflow.log_metric.call_count >= 5

    @patch('app.core.auditor.mlflow')
    def test_logs_unsupported_claims(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_fact_table
    ):
        """Should log unsupported claims for analysis."""
        # Arrange
        cover_letter = "Test"
        claim_response = {
            "content": json.dumps({"claims": ["Unsupported claim"]}),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        unsupported = {
            "content": json.dumps({
                "supported": False,
                "source": "UNSUPPORTED",
                "confidence": 0.1,
                "reasoning": "Not in CV"
            }),
            "usage": {"input_tokens": 50, "output_tokens": 30},
            "model": "gpt-4"
        }

        mock_llm_service.generate_response.side_effect = [
            claim_response,
            unsupported
        ]
        auditor = Auditor(llm_service=mock_llm_service)

        # Act
        auditor.audit(cover_letter, sample_fact_table)

        # Assert
        mock_mlflow.log_text.assert_called()
        # Verify unsupported claims were logged


class TestAuditorReportGeneration:
    """Test audit report generation."""

    def test_generate_report_calculates_metrics_correctly(self):
        """Test that report metrics are calculated correctly."""
        # Arrange
        verifications = [
            ClaimVerification(
                claim="Claim 1",
                supported=True,
                source="CV",
                confidence=0.9,
                reasoning="OK"
            ),
            ClaimVerification(
                claim="Claim 2",
                supported=True,
                source="CV",
                confidence=0.8,
                reasoning="OK"
            ),
            ClaimVerification(
                claim="Claim 3",
                supported=False,
                source="UNSUPPORTED",
                confidence=0.2,
                reasoning="Not found"
            )
        ]

        auditor = Auditor(llm_service=MagicMock())

        # Act
        report = auditor._generate_report(verifications)

        # Assert
        assert report.total_claims == 3
        assert report.supported_claims == 2
        assert report.unsupported_claims == 1
        assert abs(report.hallucination_rate - 0.333) < 0.01
        assert abs(report.overall_confidence - 0.633) < 0.01
        assert report.flagged is False  # Only 1 unsupported

    def test_generate_report_empty_verifications(self):
        """Test report generation with no verifications."""
        auditor = Auditor(llm_service=MagicMock())

        # Act
        report = auditor._generate_report([])

        # Assert
        assert report.total_claims == 0
        assert report.hallucination_rate == 0.0
        assert report.overall_confidence == 0.0
        assert report.flagged is False
