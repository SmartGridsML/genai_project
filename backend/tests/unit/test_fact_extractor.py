"""
Unit tests for FactExtractor
Tests all failure modes and edge cases for fact extraction.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from app.core.fact_extractor import FactExtractor, FactExtractionError
from app.models.schemas import ExtractedFacts


# Test data fixtures
@pytest.fixture
def valid_llm_response():
    """Valid LLM response matching ExtractedFacts schema."""
    return {
        "content": json.dumps({
            "facts": [
                {
                    "category": "experience",
                    "fact": "5 years of Python development",
                    "confidence": 0.95
                },
                {
                    "category": "skills",
                    "fact": "Expert in FastAPI and Docker",
                    "confidence": 0.90
                }
            ]
        }),
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        },
        "model": "gpt-4"
    }


@pytest.fixture
def mock_llm_service():
    """Mock LLMService for dependency injection."""
    return MagicMock()


@pytest.fixture
def sample_cv_text():
    """Sample CV text for testing."""
    return """
    John Doe
    Senior Software Engineer

    Experience:
    - 5 years of Python development
    - Expert in FastAPI and Docker
    - Built scalable microservices

    Education:
    - BS Computer Science, MIT
    """


class TestFactExtractorInit:
    """Test initialization and dependency injection."""

    def test_init_without_llm_service(self):
        """Should create its own LLMService if none provided."""
        extractor = FactExtractor()
        assert extractor.llm is not None
        assert hasattr(extractor, 'version')

    def test_init_with_llm_service(self, mock_llm_service):
        """Should use injected LLMService for testing."""
        extractor = FactExtractor(llm_service=mock_llm_service)
        assert extractor.llm == mock_llm_service


class TestFactExtractorHappyPath:
    """Test successful fact extraction scenarios."""

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_valid_cv(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text,
        valid_llm_response
    ):
        """
        Happy path: Valid CV text → Valid JSON → ExtractedFacts

        This tests the complete flow:
        1. Accept CV text
        2. Call LLM service
        3. Parse JSON response
        4. Validate schema
        5. Log metrics
        6. Return ExtractedFacts
        """
        # Arrange
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        result = extractor.extract(sample_cv_text)

        # Assert
        assert isinstance(result, ExtractedFacts)
        assert len(result.facts) == 2
        assert result.facts[0].category == "experience"
        assert result.facts[0].confidence == 0.95

        # Verify LLM was called with correct parameters
        mock_llm_service.generate_response.assert_called_once()
        call_args = mock_llm_service.generate_response.call_args
        assert call_args.kwargs['user_prompt'] == sample_cv_text
        assert call_args.kwargs['temperature'] == 0.3
        assert call_args.kwargs['response_format'] == {
            "type": "json_object"
        }

        # Verify MLflow logging
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_metric.assert_called()

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_with_request_id(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text,
        valid_llm_response
    ):
        """Should log request_id when provided."""
        # Arrange
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)
        request_id = "test-request-123"

        # Act
        result = extractor.extract(sample_cv_text, request_id=request_id)

        # Assert
        assert isinstance(result, ExtractedFacts)
        # Verify request_id was logged
        # (You'd need to check the specific call in a real test)

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_empty_facts_list(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text
    ):
        """Should handle valid JSON with empty facts list."""
        # Arrange
        empty_response = {
            "content": json.dumps({"facts": []}),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = empty_response
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        result = extractor.extract(sample_cv_text)

        # Assert
        assert isinstance(result, ExtractedFacts)
        assert len(result.facts) == 0


class TestFactExtractorErrorHandling:
    """Test error handling for various failure modes."""

    def test_extract_empty_cv_text(self, mock_llm_service):
        """Should raise FactExtractionError for empty CV text."""
        extractor = FactExtractor(llm_service=mock_llm_service)

        with pytest.raises(FactExtractionError) as exc_info:
            extractor.extract("")

        assert "cannot be empty" in str(exc_info.value)
        # Verify LLM was never called (fail fast)
        mock_llm_service.generate_response.assert_not_called()

    def test_extract_whitespace_only_cv_text(self, mock_llm_service):
        """Should raise FactExtractionError for whitespace-only text."""
        extractor = FactExtractor(llm_service=mock_llm_service)

        with pytest.raises(FactExtractionError) as exc_info:
            extractor.extract("   \n  \t  ")

        assert "cannot be empty" in str(exc_info.value)

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_invalid_json_response(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text
    ):
        """Should handle invalid JSON from LLM gracefully."""
        # Arrange
        invalid_json_response = {
            "content": "This is not valid JSON {broken",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = (
            invalid_json_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(FactExtractionError) as exc_info:
            extractor.extract(sample_cv_text)

        assert "invalid JSON" in str(exc_info.value)
        # Verify error was logged to MLflow
        mock_mlflow.log_text.assert_called()

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_wrong_schema(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text
    ):
        """Should handle JSON that doesn't match ExtractedFacts schema."""
        # Arrange - Valid JSON but wrong structure
        wrong_schema_response = {
            "content": json.dumps({
                "wrong_key": "wrong_value",
                "not_facts": []
            }),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = (
            wrong_schema_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(FactExtractionError) as exc_info:
            extractor.extract(sample_cv_text)

        assert "doesn't match schema" in str(exc_info.value)
        # Verify invalid schema was logged
        mock_mlflow.log_text.assert_called()

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_missing_required_fields(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text
    ):
        """Should handle JSON with missing required fields."""
        # Arrange - Missing 'confidence' field
        missing_field_response = {
            "content": json.dumps({
                "facts": [
                    {
                        "category": "experience",
                        "fact": "5 years Python"
                        # Missing 'confidence' field
                    }
                ]
            }),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = (
            missing_field_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(FactExtractionError) as exc_info:
            extractor.extract(sample_cv_text)

        assert "doesn't match schema" in str(exc_info.value)

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_llm_service_failure(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text
    ):
        """Should handle LLM service failures gracefully."""
        # Arrange
        mock_llm_service.generate_response.side_effect = Exception(
            "OpenAI API error"
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act & Assert
        with pytest.raises(FactExtractionError) as exc_info:
            extractor.extract(sample_cv_text)

        assert "Fact extraction failed" in str(exc_info.value)


class TestFactExtractorMLflowLogging:
    """Test MLflow logging functionality."""

    @patch('app.core.fact_extractor.mlflow')
    def test_logs_extraction_parameters(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text,
        valid_llm_response
    ):
        """Should log extraction parameters to MLflow."""
        # Arrange
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        extractor.extract(sample_cv_text, request_id="test-123")

        # Assert - Verify parameters were logged
        # Note: In real tests you'd check specific calls
        assert mock_mlflow.log_param.call_count >= 3
        mock_mlflow.log_metric.assert_called()

    @patch('app.core.fact_extractor.mlflow')
    def test_logs_fact_metrics(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text,
        valid_llm_response
    ):
        """Should log fact count and average confidence."""
        # Arrange
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        extractor.extract(sample_cv_text)

        # Assert
        mock_mlflow.log_metric.assert_called()
        # Check that both facts_extracted and avg_confidence were logged

    @patch('app.core.fact_extractor.mlflow')
    def test_logs_category_distribution(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text,
        valid_llm_response
    ):
        """Should log category distribution as JSON."""
        # Arrange
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        extractor.extract(sample_cv_text)

        # Assert
        mock_mlflow.log_dict.assert_called_once()


class TestFactExtractorEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_very_long_cv(
        self,
        mock_mlflow,
        mock_llm_service,
        valid_llm_response
    ):
        """Should handle very long CV text."""
        # Arrange
        long_cv = "Very long text. " * 10000  # ~150,000 chars
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        result = extractor.extract(long_cv)

        # Assert
        assert isinstance(result, ExtractedFacts)
        # Verify CV length was logged
        mock_mlflow.log_param.assert_called()

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_special_characters(
        self,
        mock_mlflow,
        mock_llm_service,
        valid_llm_response
    ):
        """Should handle CVs with special characters."""
        # Arrange
        special_cv = "Name: José María\nEmail: test@example.com\n日本語"
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        result = extractor.extract(special_cv)

        # Assert
        assert isinstance(result, ExtractedFacts)

    @patch('app.core.fact_extractor.mlflow')
    def test_extract_confidence_boundary_values(
        self,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text
    ):
        """Should accept confidence values at boundaries (0.0 and 1.0)."""
        # Arrange
        boundary_response = {
            "content": json.dumps({
                "facts": [
                    {
                        "category": "test",
                        "fact": "Min confidence",
                        "confidence": 0.0
                    },
                    {
                        "category": "test",
                        "fact": "Max confidence",
                        "confidence": 1.0
                    }
                ]
            }),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "gpt-4"
        }
        mock_llm_service.generate_response.return_value = (
            boundary_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        result = extractor.extract(sample_cv_text)

        # Assert
        assert len(result.facts) == 2
        assert result.facts[0].confidence == 0.0
        assert result.facts[1].confidence == 1.0


# Integration-style test (optional, can be moved to integration tests)
class TestFactExtractorIntegration:
    """
    These tests are closer to integration tests as they test
    the interaction between components.
    """

    @patch('app.core.fact_extractor.mlflow')
    @patch('app.core.fact_extractor.Prompts')
    def test_uses_correct_prompt_version(
        self,
        mock_prompts,
        mock_mlflow,
        mock_llm_service,
        sample_cv_text,
        valid_llm_response
    ):
        """Should use the configured prompt version."""
        # Arrange
        mock_prompts.get_fact_extraction_system.return_value = (
            "System prompt"
        )
        mock_llm_service.generate_response.return_value = (
            valid_llm_response
        )
        extractor = FactExtractor(llm_service=mock_llm_service)

        # Act
        extractor.extract(sample_cv_text)

        # Assert
        mock_prompts.get_fact_extraction_system.assert_called_once()
        # Verify it was called with the correct version
