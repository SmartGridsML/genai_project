"""
Unit Tests for Evaluation Suite

Tests comprehensive evaluation functionality including:
- Test case loading and validation
- Individual test execution
- Aggregate metrics calculation
- Report generation
- MLflow logging

10x Testing Principles:
- Test all code paths
- Mock external dependencies (LLM calls)
- Verify metric calculations
- Test error handling

Author: Person A - Day 5
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from backend.mlops.eval.evaluation_suite import (
    EvaluationSuite,
    TestCase,
    TestCaseResult,
    AggregateMetrics
)
from app.models.schemas import ExtractedFacts, KeyFact, AuditReport, ClaimVerification


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_test_cases_file():
    """Create a temporary test cases JSON file."""
    test_data = {
        "test_cases": [
            {
                "id": "test1",
                "name": "Test Case 1",
                "description": "First test",
                "cv_text": "Sample CV with 5 years Python experience",
                "job_description": "Python developer needed",
                "expected_hallucination_rate": 0.0,
                "expected_facts_count": 5,
                "expected_claims_count": 8,
                "notes": "Happy path test"
            },
            {
                "id": "test2",
                "name": "Test Case 2",
                "description": "Second test",
                "cv_text": "Junior developer fresh graduate",
                "job_description": "Entry level position",
                "expected_hallucination_rate": 0.05,
                "expected_facts_count": 3,
                "expected_claims_count": 5,
                "notes": "Edge case test"
            }
        ],
        "metadata": {
            "version": "1.0",
            "total_test_cases": 2
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(test_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def mock_facts():
    """Mock ExtractedFacts response."""
    return ExtractedFacts(
        facts=[
            KeyFact(category="experience", fact="5 years Python", confidence=0.9),
            KeyFact(category="skills", fact="FastAPI expert", confidence=0.85),
            KeyFact(category="education", fact="BS CS MIT", confidence=0.95)
        ]
    )


@pytest.fixture
def mock_audit_report():
    """Mock AuditReport response."""
    return AuditReport(
        verifications=[
            ClaimVerification(
                claim="5 years Python experience",
                supported=True,
                source="CV fact: 5 years Python",
                confidence=1.0,
                reasoning="Exact match"
            ),
            ClaimVerification(
                claim="FastAPI expert",
                supported=True,
                source="CV fact: FastAPI expert",
                confidence=0.9,
                reasoning="Direct support"
            )
        ],
        total_claims=2,
        supported_claims=2,
        unsupported_claims=0,
        hallucination_rate=0.0,
        flagged=False,
        overall_confidence=0.95
    )


# =============================================================================
# TEST CASES
# =============================================================================

class TestEvaluationSuiteInit:
    """Test EvaluationSuite initialization."""

    def test_init_loads_test_cases(self, temp_test_cases_file):
        """Should load test cases from JSON file."""
        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        assert len(suite.test_cases) == 2
        assert suite.test_cases[0].id == "test1"
        assert suite.test_cases[1].id == "test2"

    def test_init_creates_output_directory(self, temp_test_cases_file):
        """Should create output directory if it doesn't exist."""
        output_dir = Path(tempfile.mkdtemp()) / "new_dir"

        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=str(output_dir)
        )

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_init_fails_on_missing_file(self):
        """Should raise error if test cases file not found."""
        with pytest.raises(FileNotFoundError):
            EvaluationSuite(
                test_cases_path="/nonexistent/path.json",
                output_dir=tempfile.mkdtemp()
            )

    def test_init_fails_on_invalid_json(self):
        """Should raise error if JSON is invalid."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("{invalid json")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                EvaluationSuite(
                    test_cases_path=temp_path,
                    output_dir=tempfile.mkdtemp()
                )
        finally:
            Path(temp_path).unlink()


class TestTestCaseModel:
    """Test TestCase Pydantic model."""

    def test_valid_test_case(self):
        """Should create valid test case."""
        tc = TestCase(
            id="test1",
            name="Test",
            description="Description",
            cv_text="CV content",
            job_description="JD content",
            expected_hallucination_rate=0.05,
            expected_facts_count=10,
            expected_claims_count=15,
            notes="Notes"
        )

        assert tc.id == "test1"
        assert tc.expected_hallucination_rate == 0.05

    def test_validation_hallucination_rate_bounds(self):
        """Should validate hallucination rate is between 0 and 1."""
        # Valid: 0.0
        tc1 = TestCase(
            id="test1",
            name="Test",
            description="Desc",
            cv_text="CV",
            job_description="JD",
            expected_hallucination_rate=0.0,
            expected_facts_count=5,
            expected_claims_count=5
        )
        assert tc1.expected_hallucination_rate == 0.0

        # Valid: 1.0
        tc2 = TestCase(
            id="test2",
            name="Test",
            description="Desc",
            cv_text="CV",
            job_description="JD",
            expected_hallucination_rate=1.0,
            expected_facts_count=5,
            expected_claims_count=5
        )
        assert tc2.expected_hallucination_rate == 1.0

        # Invalid: >1.0
        with pytest.raises(ValueError):
            TestCase(
                id="test3",
                name="Test",
                description="Desc",
                cv_text="CV",
                job_description="JD",
                expected_hallucination_rate=1.5,
                expected_facts_count=5,
                expected_claims_count=5
            )


class TestRunSingleTest:
    """Test running a single test case."""

    @patch('backend.mlops.eval.evaluation_suite.fact_extractor')
    @patch('backend.mlops.eval.evaluation_suite.analyze_job_description', new_callable=AsyncMock)
    @patch('backend.mlops.eval.evaluation_suite.generate_cover_letter', new_callable=AsyncMock)
    @patch('backend.mlops.eval.evaluation_suite.auditor')
    def test_successful_test_execution(
        self,
        mock_auditor,
        mock_generate,
        mock_analyze_jd,
        mock_fact_extractor,
        temp_test_cases_file,
        mock_facts,
        mock_audit_report
    ):
        """Should successfully execute a test case."""
        # Setup mocks
        mock_fact_extractor.extract.return_value = mock_facts
        mock_analyze_jd.return_value = {"summary": "Good match"}

        # Mock the async generate_cover_letter to return a string
        mock_generate.return_value = "This is a generated cover letter with about fifteen words in it for testing purposes only."
        mock_auditor.audit.return_value = mock_audit_report

        # Create suite and load test case
        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        # Run single test
        result = suite._run_single_test(suite.test_cases[0])

        # Assertions
        assert result.success is True
        assert result.error is None
        assert result.facts_extracted == 3  # From mock_facts
        assert result.claims_extracted == 2  # From mock_audit_report
        assert result.hallucination_rate == 0.0
        assert result.flagged is False
        assert result.total_latency_ms > 0

    @patch('backend.mlops.eval.evaluation_suite.fact_extractor')
    def test_failed_test_execution(
        self,
        mock_fact_extractor,
        temp_test_cases_file
    ):
        """Should handle test failures gracefully."""
        # Setup mock to raise error
        mock_fact_extractor.extract.side_effect = Exception("LLM API error")

        # Create suite
        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        # Run single test
        result = suite._run_single_test(suite.test_cases[0])

        # Assertions
        assert result.success is False
        assert result.error == "LLM API error"
        assert result.facts_extracted == 0
        assert result.hallucination_rate == 1.0
        assert result.flagged is True


class TestAggregateMetrics:
    """Test aggregate metrics calculation."""

    def test_calculate_aggregate_all_successful(self, temp_test_cases_file):
        """Should calculate correct aggregate when all tests succeed."""
        # Create mock results
        results = [
            TestCaseResult(
                test_id="test1",
                test_name="Test 1",
                timestamp=datetime.utcnow().isoformat(),
                duration_seconds=2.5,
                success=True,
                facts_extracted=5,
                claims_extracted=8,
                cover_letter_length=200,
                cover_letter_word_count=50,
                hallucination_rate=0.0,
                supported_claims=8,
                unsupported_claims=0,
                flagged=False,
                overall_confidence=0.95,
                fact_extraction_latency_ms=500,
                jd_analysis_latency_ms=300,
                generation_latency_ms=1000,
                audit_latency_ms=700,
                total_latency_ms=2500,
                total_tokens_used=500,
                estimated_cost_usd=0.015,
                facts_count_delta=0,
                claims_count_delta=0,
                hallucination_rate_delta=0.0
            ),
            TestCaseResult(
                test_id="test2",
                test_name="Test 2",
                timestamp=datetime.utcnow().isoformat(),
                duration_seconds=3.0,
                success=True,
                facts_extracted=3,
                claims_extracted=5,
                cover_letter_length=150,
                cover_letter_word_count=40,
                hallucination_rate=0.2,
                supported_claims=4,
                unsupported_claims=1,
                flagged=False,
                overall_confidence=0.80,
                fact_extraction_latency_ms=600,
                jd_analysis_latency_ms=400,
                generation_latency_ms=1200,
                audit_latency_ms=800,
                total_latency_ms=3000,
                total_tokens_used=450,
                estimated_cost_usd=0.014,
                facts_count_delta=0,
                claims_count_delta=0,
                hallucination_rate_delta=0.0
            )
        ]

        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        aggregate = suite._calculate_aggregate_metrics(results)

        # Assertions
        assert aggregate.total_tests == 2
        assert aggregate.successful_tests == 2
        assert aggregate.failed_tests == 0
        assert aggregate.success_rate == 1.0

        assert aggregate.avg_hallucination_rate == 0.1  # (0.0 + 0.2) / 2
        assert aggregate.median_hallucination_rate == 0.1
        assert aggregate.max_hallucination_rate == 0.2

        assert aggregate.avg_confidence == 0.875  # (0.95 + 0.80) / 2

        assert aggregate.avg_total_latency_ms == 2750  # (2500 + 3000) / 2
        assert aggregate.p50_latency_ms == 2750  # Median of [2500, 3000] = (2500 + 3000) / 2

        assert aggregate.total_cost_usd == pytest.approx(0.029)  # 0.015 + 0.014
        assert aggregate.avg_cost_per_request_usd == pytest.approx(0.0145)

    def test_calculate_aggregate_with_failures(self, temp_test_cases_file):
        """Should handle mix of successful and failed tests."""
        results = [
            TestCaseResult(
                test_id="test1",
                test_name="Test 1",
                timestamp=datetime.utcnow().isoformat(),
                duration_seconds=2.5,
                success=True,
                facts_extracted=5,
                claims_extracted=8,
                cover_letter_length=200,
                cover_letter_word_count=50,
                hallucination_rate=0.0,
                supported_claims=8,
                unsupported_claims=0,
                flagged=False,
                overall_confidence=0.95,
                fact_extraction_latency_ms=500,
                jd_analysis_latency_ms=300,
                generation_latency_ms=1000,
                audit_latency_ms=700,
                total_latency_ms=2500,
                total_tokens_used=500,
                estimated_cost_usd=0.015,
                facts_count_delta=0,
                claims_count_delta=0,
                hallucination_rate_delta=0.0
            ),
            TestCaseResult(
                test_id="test2",
                test_name="Test 2",
                timestamp=datetime.utcnow().isoformat(),
                duration_seconds=1.0,
                success=False,
                error="API error",
                facts_extracted=0,
                claims_extracted=0,
                cover_letter_length=0,
                cover_letter_word_count=0,
                hallucination_rate=1.0,
                supported_claims=0,
                unsupported_claims=0,
                flagged=True,
                overall_confidence=0.0,
                fact_extraction_latency_ms=0,
                jd_analysis_latency_ms=0,
                generation_latency_ms=0,
                audit_latency_ms=0,
                total_latency_ms=0,
                total_tokens_used=0,
                estimated_cost_usd=0.0,
                facts_count_delta=0,
                claims_count_delta=0,
                hallucination_rate_delta=0.0
            )
        ]

        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        aggregate = suite._calculate_aggregate_metrics(results)

        # Should only use successful tests for quality metrics
        assert aggregate.total_tests == 2
        assert aggregate.successful_tests == 1
        assert aggregate.failed_tests == 1
        assert aggregate.success_rate == 0.5

        assert aggregate.avg_hallucination_rate == 0.0  # Only from successful
        assert aggregate.avg_confidence == 0.95

    def test_calculate_aggregate_no_successful(self, temp_test_cases_file):
        """Should handle case where all tests fail."""
        results = [
            TestCaseResult(
                test_id="test1",
                test_name="Test 1",
                timestamp=datetime.utcnow().isoformat(),
                duration_seconds=1.0,
                success=False,
                error="Error",
                facts_extracted=0,
                claims_extracted=0,
                cover_letter_length=0,
                cover_letter_word_count=0,
                hallucination_rate=1.0,
                supported_claims=0,
                unsupported_claims=0,
                flagged=True,
                overall_confidence=0.0,
                fact_extraction_latency_ms=0,
                jd_analysis_latency_ms=0,
                generation_latency_ms=0,
                audit_latency_ms=0,
                total_latency_ms=0,
                total_tokens_used=0,
                estimated_cost_usd=0.0,
                facts_count_delta=0,
                claims_count_delta=0,
                hallucination_rate_delta=0.0
            )
        ]

        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        aggregate = suite._calculate_aggregate_metrics(results)

        assert aggregate.success_rate == 0.0
        assert aggregate.total_cost_usd == 0.0


class TestReportGeneration:
    """Test evaluation report generation."""

    def test_generate_report_structure(self, temp_test_cases_file):
        """Should generate report with correct structure."""
        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        # Create mock results
        results = []
        aggregate = AggregateMetrics(
            total_tests=2,
            successful_tests=2,
            failed_tests=0,
            success_rate=1.0,
            avg_hallucination_rate=0.03,
            median_hallucination_rate=0.03,
            max_hallucination_rate=0.05,
            avg_confidence=0.90,
            flagged_rate=0.0,
            avg_total_latency_ms=2500,
            p50_latency_ms=2500,
            p95_latency_ms=2800,
            p99_latency_ms=2900,
            total_cost_usd=0.030,
            avg_cost_per_request_usd=0.015,
            total_tokens=1000,
            avg_facts_delta=0.0,
            avg_claims_delta=0.0,
            avg_hallucination_delta=0.0
        )

        report = suite._generate_report(results, aggregate)

        # Check structure
        assert 'metadata' in report
        assert 'aggregate_metrics' in report
        assert 'individual_results' in report
        assert 'summary' in report

        # Check metadata
        assert 'timestamp' in report['metadata']
        assert report['metadata']['total_test_cases'] == 2

        # Check summary thresholds
        assert 'passed_quality_threshold' in report['summary']
        assert 'passed_latency_threshold' in report['summary']
        assert 'passed_cost_threshold' in report['summary']
        assert 'overall_pass' in report['summary']

        # Should pass quality threshold (<5% hallucination)
        assert report['summary']['passed_quality_threshold'] is True

        # Should pass latency threshold (<30s p95)
        assert report['summary']['passed_latency_threshold'] is True

    def test_report_saved_to_file(self, temp_test_cases_file):
        """Should save report to file."""
        output_dir = Path(tempfile.mkdtemp())

        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=str(output_dir)
        )

        results = []
        aggregate = AggregateMetrics(
            total_tests=1,
            successful_tests=1,
            failed_tests=0,
            success_rate=1.0,
            avg_hallucination_rate=0.0,
            median_hallucination_rate=0.0,
            max_hallucination_rate=0.0,
            avg_confidence=1.0,
            flagged_rate=0.0,
            avg_total_latency_ms=1000,
            p50_latency_ms=1000,
            p95_latency_ms=1000,
            p99_latency_ms=1000,
            total_cost_usd=0.01,
            avg_cost_per_request_usd=0.01,
            total_tokens=500,
            avg_facts_delta=0.0,
            avg_claims_delta=0.0,
            avg_hallucination_delta=0.0
        )

        suite._generate_report(results, aggregate)

        # Check file was created
        report_files = list(output_dir.glob("evaluation_report_*.json"))
        assert len(report_files) == 1

        # Verify file contents
        with open(report_files[0], 'r') as f:
            saved_report = json.load(f)

        assert 'aggregate_metrics' in saved_report


class TestMLflowLogging:
    """Test MLflow integration."""

    @patch('backend.mlops.eval.evaluation_suite.mlflow')
    def test_logs_to_mlflow(self, mock_mlflow, temp_test_cases_file):
        """Should log metrics to MLflow."""
        suite = EvaluationSuite(
            test_cases_path=temp_test_cases_file,
            output_dir=tempfile.mkdtemp()
        )

        results = []
        aggregate = AggregateMetrics(
            total_tests=2,
            successful_tests=2,
            failed_tests=0,
            success_rate=1.0,
            avg_hallucination_rate=0.05,
            median_hallucination_rate=0.05,
            max_hallucination_rate=0.05,
            avg_confidence=0.90,
            flagged_rate=0.0,
            avg_total_latency_ms=2500,
            p50_latency_ms=2500,
            p95_latency_ms=2800,
            p99_latency_ms=2900,
            total_cost_usd=0.030,
            avg_cost_per_request_usd=0.015,
            total_tokens=1000,
            avg_facts_delta=0.0,
            avg_claims_delta=0.0,
            avg_hallucination_delta=0.0
        )

        suite._log_to_mlflow(results, aggregate)

        # Verify MLflow was called
        mock_mlflow.start_run.assert_called_once()
        assert mock_mlflow.log_metric.call_count >= 10  # Many metrics logged
        mock_mlflow.log_text.assert_called_once()  # Results artifact logged
