"""
Evaluation Suite for Cover Letter Generation Pipeline

This module provides comprehensive evaluation of the entire pipeline:
- Fact extraction accuracy
- Cover letter generation quality
- Hallucination detection effectiveness
- Performance metrics (latency, cost)

10x Principles:
1. Quantitative Metrics: Measure everything
2. Reproducibility: Same inputs = same metrics
3. Comprehensive Coverage: Test all components
4. Actionable Insights: Generate clear reports

Author: Person A - Day 5
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev
from datetime import datetime

import mlflow
from pydantic import BaseModel, Field

from backend.app.core.fact_extractor import fact_extractor
from backend.app.core.jd_analyzer import analyze_job_description
from backend.app.core.cover_letter_gen import generate_cover_letter
from backend.app.core.auditor import auditor
from backend.app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class TestCase(BaseModel):
    """Single test case from test_cases.json"""
    id: str
    name: str
    description: str
    cv_text: str
    job_description: str
    expected_hallucination_rate: float = Field(ge=0.0, le=1.0)
    expected_facts_count: int = Field(ge=0)
    expected_claims_count: int = Field(ge=0)
    notes: str = ""


class TestCaseResult(BaseModel):
    """Results for a single test case"""
    test_id: str
    test_name: str

    # Execution metadata
    timestamp: str
    duration_seconds: float
    success: bool
    error: Optional[str] = None

    # Pipeline outputs
    facts_extracted: int
    claims_extracted: int
    cover_letter_length: int
    cover_letter_word_count: int

    # Quality metrics
    hallucination_rate: float
    supported_claims: int
    unsupported_claims: int
    flagged: bool
    overall_confidence: float

    # Performance metrics
    fact_extraction_latency_ms: float
    jd_analysis_latency_ms: float
    generation_latency_ms: float
    audit_latency_ms: float
    total_latency_ms: float

    # Cost metrics
    total_tokens_used: int
    estimated_cost_usd: float

    # Comparison to expected
    facts_count_delta: int
    claims_count_delta: int
    hallucination_rate_delta: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all test cases"""
    # Summary
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float

    # Quality metrics (averages)
    avg_hallucination_rate: float
    median_hallucination_rate: float
    max_hallucination_rate: float

    avg_confidence: float
    flagged_rate: float

    # Performance metrics
    avg_total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Cost metrics
    total_cost_usd: float
    avg_cost_per_request_usd: float
    total_tokens: int

    # Accuracy metrics
    avg_facts_delta: float
    avg_claims_delta: float
    avg_hallucination_delta: float


class EvaluationSuite:
    """
    Comprehensive evaluation suite for the cover letter pipeline.

    10x Engineering:
    - Automated testing of production quality
    - Quantitative metrics for every aspect
    - Clear reporting for stakeholders
    - MLflow integration for tracking trends
    """

    # Cost estimation (OpenAI GPT-4 pricing as of 2024)
    COST_PER_1K_INPUT_TOKENS = 0.03  # $0.03 per 1K input tokens
    COST_PER_1K_OUTPUT_TOKENS = 0.06  # $0.06 per 1K output tokens

    def __init__(
        self,
        test_cases_path: str = "backend/mlops/eval/test_cases.json",
        output_dir: str = "backend/mlops/eval/reports"
    ):
        """
        Initialize evaluation suite.

        Args:
            test_cases_path: Path to test_cases.json
            output_dir: Directory for evaluation reports
        """
        self.test_cases_path = Path(test_cases_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load test cases
        self.test_cases = self._load_test_cases()
        logger.info(f"Loaded {len(self.test_cases)} test cases")

    def _load_test_cases(self) -> List[TestCase]:
        """Load test cases from JSON file."""
        try:
            with open(self.test_cases_path, 'r') as f:
                data = json.load(f)

            test_cases = []
            for tc_data in data['test_cases']:
                test_cases.append(TestCase(**tc_data))

            return test_cases

        except FileNotFoundError:
            logger.error(f"Test cases file not found: {self.test_cases_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test cases file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            raise

    def run_evaluation(
        self,
        test_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on all or selected test cases.

        Args:
            test_ids: Optional list of test IDs to run (runs all if None)

        Returns:
            Complete evaluation results with aggregate metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING EVALUATION SUITE")
        logger.info("=" * 80)

        # Filter test cases if specific IDs provided
        if test_ids:
            test_cases = [tc for tc in self.test_cases if tc.id in test_ids]
            logger.info(f"Running {len(test_cases)} selected test cases")
        else:
            test_cases = self.test_cases
            logger.info(f"Running all {len(test_cases)} test cases")

        # Run each test case
        results = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n[{i}/{len(test_cases)}] Running: {test_case.name}")
            result = self._run_single_test(test_case)
            results.append(result)

            # Log result summary
            if result.success:
                logger.info(
                    f"✓ Success | Hallucination: {result.hallucination_rate:.1%} | "
                    f"Latency: {result.total_latency_ms:.0f}ms | "
                    f"Cost: ${result.estimated_cost_usd:.4f}"
                )
            else:
                logger.error(f"✗ Failed | Error: {result.error}")

        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results)

        # Generate report
        report = self._generate_report(results, aggregate)

        # Log to MLflow
        self._log_to_mlflow(results, aggregate)

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Success Rate: {aggregate.success_rate:.1%}")
        logger.info(f"Avg Hallucination Rate: {aggregate.avg_hallucination_rate:.1%}")
        logger.info(f"Avg Latency: {aggregate.avg_total_latency_ms:.0f}ms")
        logger.info(f"Total Cost: ${aggregate.total_cost_usd:.2f}")

        return report

    def _run_single_test(self, test_case: TestCase) -> TestCaseResult:
        """
        Run a single test case through the entire pipeline.

        Returns:
            TestCaseResult with all metrics
        """
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat()

        # Initialize metrics
        total_tokens = 0
        latencies = {}

        try:
            # Initialize LLM service once
            llm_service = LLMService()

            # Step 1: Extract facts from CV
            logger.debug("Step 1: Extracting facts from CV")
            fact_start = time.time()
            facts = fact_extractor.extract(test_case.cv_text)
            latencies['fact_extraction'] = (time.time() - fact_start) * 1000

            facts_count = len(facts.facts)
            logger.debug(f"Extracted {facts_count} facts")

            # Step 2: Analyze job description
            logger.debug("Step 2: Analyzing job description")
            jd_start = time.time()
            job_analysis = asyncio.run(
                analyze_job_description(llm_service, test_case.job_description)
            )
            latencies['jd_analysis'] = (time.time() - jd_start) * 1000

            # Step 3: Generate cover letter
            logger.debug("Step 3: Generating cover letter")
            gen_start = time.time()

            # Convert facts to dict format expected by generator
            facts_dict = {
                "facts": [f.model_dump() for f in facts.facts]
            }
            job_dict = job_analysis

            # Generate cover letter (async)
            cover_letter = asyncio.run(
                generate_cover_letter(
                    llm_service,
                    facts_dict,
                    job_dict,
                    tone="professional"
                )
            )

            latencies['generation'] = (time.time() - gen_start) * 1000

            letter_length = len(cover_letter)
            word_count = len(cover_letter.split())
            logger.debug(f"Generated {word_count} words")

            # Step 4: Audit for hallucinations
            logger.debug("Step 4: Auditing for hallucinations")
            audit_start = time.time()
            audit_report = auditor.audit(
                cover_letter,
                facts,
                request_id=f"eval_{test_case.id}"
            )
            latencies['audit'] = (time.time() - audit_start) * 1000

            # Calculate total latency
            total_duration = time.time() - start_time
            total_latency_ms = sum(latencies.values())

            # Estimate cost (simplified - would need actual token counts from LLM responses)
            # For now, use rough estimation based on text lengths
            estimated_tokens = (
                len(test_case.cv_text.split()) * 1.3 +  # CV tokens
                len(test_case.job_description.split()) * 1.3 +  # JD tokens
                len(cover_letter.split()) * 1.3 +  # Generated tokens
                audit_report.total_claims * 100  # Audit tokens (rough estimate)
            )
            total_tokens = int(estimated_tokens)

            estimated_cost = (
                (total_tokens * 0.7 * self.COST_PER_1K_INPUT_TOKENS / 1000) +
                (total_tokens * 0.3 * self.COST_PER_1K_OUTPUT_TOKENS / 1000)
            )

            # Calculate deltas from expected
            facts_delta = facts_count - test_case.expected_facts_count
            claims_delta = audit_report.total_claims - test_case.expected_claims_count
            hallucination_delta = (
                audit_report.hallucination_rate -
                test_case.expected_hallucination_rate
            )

            # Create result
            result = TestCaseResult(
                test_id=test_case.id,
                test_name=test_case.name,
                timestamp=timestamp,
                duration_seconds=total_duration,
                success=True,
                error=None,
                facts_extracted=facts_count,
                claims_extracted=audit_report.total_claims,
                cover_letter_length=letter_length,
                cover_letter_word_count=word_count,
                hallucination_rate=audit_report.hallucination_rate,
                supported_claims=audit_report.supported_claims,
                unsupported_claims=audit_report.unsupported_claims,
                flagged=audit_report.flagged,
                overall_confidence=audit_report.overall_confidence,
                fact_extraction_latency_ms=latencies['fact_extraction'],
                jd_analysis_latency_ms=latencies['jd_analysis'],
                generation_latency_ms=latencies['generation'],
                audit_latency_ms=latencies['audit'],
                total_latency_ms=total_latency_ms,
                total_tokens_used=total_tokens,
                estimated_cost_usd=estimated_cost,
                facts_count_delta=facts_delta,
                claims_count_delta=claims_delta,
                hallucination_rate_delta=hallucination_delta
            )

            return result

        except Exception as e:
            # Test failed - capture error
            logger.error(f"Test case failed: {e}", exc_info=True)

            total_duration = time.time() - start_time

            return TestCaseResult(
                test_id=test_case.id,
                test_name=test_case.name,
                timestamp=timestamp,
                duration_seconds=total_duration,
                success=False,
                error=str(e),
                facts_extracted=0,
                claims_extracted=0,
                cover_letter_length=0,
                cover_letter_word_count=0,
                hallucination_rate=1.0,
                supported_claims=0,
                unsupported_claims=0,
                flagged=True,
                overall_confidence=0.0,
                fact_extraction_latency_ms=0.0,
                jd_analysis_latency_ms=0.0,
                generation_latency_ms=0.0,
                audit_latency_ms=0.0,
                total_latency_ms=0.0,
                total_tokens_used=0,
                estimated_cost_usd=0.0,
                facts_count_delta=0,
                claims_count_delta=0,
                hallucination_rate_delta=0.0
            )

    def _calculate_aggregate_metrics(
        self,
        results: List[TestCaseResult]
    ) -> AggregateMetrics:
        """Calculate aggregate metrics across all test results."""

        # Filter successful tests for quality metrics
        successful = [r for r in results if r.success]

        if not successful:
            logger.warning("No successful tests - cannot calculate metrics")
            return AggregateMetrics(
                total_tests=len(results),
                successful_tests=0,
                failed_tests=len(results),
                success_rate=0.0,
                avg_hallucination_rate=1.0,
                median_hallucination_rate=1.0,
                max_hallucination_rate=1.0,
                avg_confidence=0.0,
                flagged_rate=1.0,
                avg_total_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                total_cost_usd=0.0,
                avg_cost_per_request_usd=0.0,
                total_tokens=0,
                avg_facts_delta=0.0,
                avg_claims_delta=0.0,
                avg_hallucination_delta=0.0
            )

        # Quality metrics
        hallucination_rates = [r.hallucination_rate for r in successful]
        confidences = [r.overall_confidence for r in successful]
        flagged_count = sum(1 for r in successful if r.flagged)

        # Latency metrics
        latencies = [r.total_latency_ms for r in successful]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        # Calculate percentiles properly
        # P50 is the median - use statistics.median for accuracy
        p50 = median(latencies) if latencies else 0
        # For P95 and P99, use index-based calculation with proper rounding
        p95_index = min(int(n * 0.95 + 0.5), n - 1) if n > 0 else 0
        p99_index = min(int(n * 0.99 + 0.5), n - 1) if n > 0 else 0
        p95 = latencies_sorted[p95_index] if n > 0 else 0
        p99 = latencies_sorted[p99_index] if n > 0 else 0

        # Cost metrics
        total_cost = sum(r.estimated_cost_usd for r in successful)
        total_tokens = sum(r.total_tokens_used for r in successful)

        # Accuracy metrics
        facts_deltas = [r.facts_count_delta for r in successful]
        claims_deltas = [r.claims_count_delta for r in successful]
        hallucination_deltas = [r.hallucination_rate_delta for r in successful]

        return AggregateMetrics(
            total_tests=len(results),
            successful_tests=len(successful),
            failed_tests=len(results) - len(successful),
            success_rate=len(successful) / len(results) if results else 0.0,
            avg_hallucination_rate=mean(hallucination_rates),
            median_hallucination_rate=median(hallucination_rates),
            max_hallucination_rate=max(hallucination_rates),
            avg_confidence=mean(confidences),
            flagged_rate=flagged_count / len(successful) if successful else 0.0,
            avg_total_latency_ms=mean(latencies),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            total_cost_usd=total_cost,
            avg_cost_per_request_usd=total_cost / len(successful) if successful else 0.0,
            total_tokens=total_tokens,
            avg_facts_delta=mean(facts_deltas) if facts_deltas else 0.0,
            avg_claims_delta=mean(claims_deltas) if claims_deltas else 0.0,
            avg_hallucination_delta=mean(hallucination_deltas) if hallucination_deltas else 0.0
        )

    def _generate_report(
        self,
        results: List[TestCaseResult],
        aggregate: AggregateMetrics
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        report = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "test_cases_file": str(self.test_cases_path),
                "total_test_cases": len(self.test_cases)
            },
            "aggregate_metrics": asdict(aggregate),
            "individual_results": [r.model_dump() for r in results],
            "summary": {
                "passed_quality_threshold": aggregate.avg_hallucination_rate < 0.05,
                "passed_latency_threshold": aggregate.p95_latency_ms < 30000,
                "passed_cost_threshold": aggregate.avg_cost_per_request_usd < 0.10,
                "overall_pass": (
                    aggregate.success_rate >= 0.95 and
                    aggregate.avg_hallucination_rate < 0.05 and
                    aggregate.p95_latency_ms < 30000
                )
            }
        }

        # Save report to file
        report_filename = f"evaluation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {report_path}")

        return report

    def _log_to_mlflow(
        self,
        results: List[TestCaseResult],
        aggregate: AggregateMetrics
    ):
        """Log evaluation metrics to MLflow."""

        with mlflow.start_run(run_name="evaluation_suite", nested=True):
            # Log aggregate metrics
            mlflow.log_metric("total_tests", aggregate.total_tests)
            mlflow.log_metric("successful_tests", aggregate.successful_tests)
            mlflow.log_metric("failed_tests", aggregate.failed_tests)
            mlflow.log_metric("success_rate", aggregate.success_rate)

            mlflow.log_metric("avg_hallucination_rate", aggregate.avg_hallucination_rate)
            mlflow.log_metric("median_hallucination_rate", aggregate.median_hallucination_rate)
            mlflow.log_metric("max_hallucination_rate", aggregate.max_hallucination_rate)
            mlflow.log_metric("avg_confidence", aggregate.avg_confidence)
            mlflow.log_metric("flagged_rate", aggregate.flagged_rate)

            mlflow.log_metric("avg_latency_ms", aggregate.avg_total_latency_ms)
            mlflow.log_metric("p50_latency_ms", aggregate.p50_latency_ms)
            mlflow.log_metric("p95_latency_ms", aggregate.p95_latency_ms)
            mlflow.log_metric("p99_latency_ms", aggregate.p99_latency_ms)

            mlflow.log_metric("total_cost_usd", aggregate.total_cost_usd)
            mlflow.log_metric("avg_cost_per_request_usd", aggregate.avg_cost_per_request_usd)
            mlflow.log_metric("total_tokens", aggregate.total_tokens)

            # Log accuracy deltas
            mlflow.log_metric("avg_facts_delta", aggregate.avg_facts_delta)
            mlflow.log_metric("avg_claims_delta", aggregate.avg_claims_delta)
            mlflow.log_metric("avg_hallucination_delta", aggregate.avg_hallucination_delta)

            # Log individual results as artifact
            results_json = json.dumps(
                [r.model_dump() for r in results],
                indent=2
            )
            mlflow.log_text(results_json, "detailed_results.json")

            logger.info("Metrics logged to MLflow")


# CLI interface
def main():
    """Run evaluation suite from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument(
        "--test-ids",
        nargs="+",
        help="Specific test IDs to run (runs all if not specified)"
    )
    parser.add_argument(
        "--test-cases",
        default="backend/mlops/eval/test_cases.json",
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="backend/mlops/eval/reports",
        help="Output directory for reports"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run evaluation
    suite = EvaluationSuite(
        test_cases_path=args.test_cases,
        output_dir=args.output_dir
    )

    report = suite.run_evaluation(test_ids=args.test_ids)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall Pass: {report['summary']['overall_pass']}")
    print(f"Success Rate: {report['aggregate_metrics']['success_rate']:.1%}")
    print(f"Avg Hallucination Rate: {report['aggregate_metrics']['avg_hallucination_rate']:.1%}")
    print(f"P95 Latency: {report['aggregate_metrics']['p95_latency_ms']:.0f}ms")
    print(f"Total Cost: ${report['aggregate_metrics']['total_cost_usd']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
