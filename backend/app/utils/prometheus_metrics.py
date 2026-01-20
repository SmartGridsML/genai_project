"""
Prometheus Metrics for Cover Letter Generation Pipeline

This module defines and exports Prometheus metrics for monitoring
the application in production.

10x Principles:
1. Observability: Instrument every critical path
2. SLO-driven: Track metrics that matter for SLAs
3. Actionable: Metrics that drive alerts and decisions
4. Low-overhead: Efficient metric collection

Metrics Categories:
- Request metrics: Total requests, success/failure rates
- Latency metrics: Request duration at various percentiles
- Business metrics: Hallucination rate, quality scores
- Cost metrics: LLM token usage and costs

Author: Person A - Day 5
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from functools import wraps
from time import time
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST METRICS
# =============================================================================

# Total application requests
application_requests_total = Counter(
    'application_requests_total',
    'Total number of cover letter generation requests',
    ['endpoint', 'status']  # Labels: endpoint name, success/failure
)

# Request duration histogram
application_latency_seconds = Histogram(
    'application_latency_seconds',
    'Cover letter generation request duration in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]  # Buckets in seconds
)

# Current in-flight requests
application_requests_in_progress = Gauge(
    'application_requests_in_progress',
    'Number of requests currently being processed',
    ['endpoint']
)


# =============================================================================
# LLM-SPECIFIC METRICS
# =============================================================================

# LLM API call counts
llm_api_calls_total = Counter(
    'llm_api_calls_total',
    'Total number of LLM API calls',
    ['operation', 'model', 'status']  # operation: fact_extraction, generation, audit, etc.
)

# LLM latency
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM API call duration in seconds',
    ['operation', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
)

# Token usage
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens consumed',
    ['operation', 'token_type']  # token_type: input, output
)

# Cost tracking
llm_cost_dollars = Counter(
    'llm_cost_dollars',
    'Total LLM API costs in USD',
    ['operation', 'model']
)


# =============================================================================
# QUALITY METRICS
# =============================================================================

# Hallucination rate
hallucination_rate = Histogram(
    'hallucination_rate',
    'Hallucination rate per request (0.0 to 1.0)',
    buckets=[0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
)

# Audit flagged requests
audit_flagged_total = Counter(
    'audit_flagged_total',
    'Total number of requests flagged by auditor',
    ['reason']  # reason: high_hallucination, low_confidence, etc.
)

# Confidence scores
confidence_score = Histogram(
    'confidence_score',
    'Overall confidence score per request (0.0 to 1.0)',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Claims verification
claims_total = Counter(
    'claims_total',
    'Total number of claims processed',
    ['supported']  # supported: true/false
)

# Facts extraction
facts_extracted = Histogram(
    'facts_extracted',
    'Number of facts extracted per CV',
    buckets=[0, 5, 10, 15, 20, 30, 50, 100]
)


# =============================================================================
# ERROR METRICS
# =============================================================================

# Errors by type
application_errors_total = Counter(
    'application_errors_total',
    'Total number of errors',
    ['error_type', 'component']  # component: fact_extractor, auditor, etc.
)

# Validation failures
validation_failures_total = Counter(
    'validation_failures_total',
    'Total number of validation failures',
    ['validation_type']  # input, schema, business_logic, etc.
)


# =============================================================================
# SYSTEM METRICS
# =============================================================================

# Application info
application_info = Info(
    'application',
    'Application version and metadata'
)

# Set application info (call this at startup)
application_info.info({
    'version': '1.0.0',
    'environment': 'production',  # or from config
    'component': 'cover_letter_service'
})


# =============================================================================
# UTILITY DECORATORS
# =============================================================================

def track_request_metrics(endpoint: str):
    """
    Decorator to track request metrics automatically.

    Usage:
        @track_request_metrics("generate_cover_letter")
        def generate_cover_letter(cv, job):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Increment in-progress gauge
            application_requests_in_progress.labels(endpoint=endpoint).inc()

            # Track latency
            start_time = time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success
                application_requests_total.labels(
                    endpoint=endpoint,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                # Record failure
                application_requests_total.labels(
                    endpoint=endpoint,
                    status='failure'
                ).inc()

                # Track error type
                application_errors_total.labels(
                    error_type=type(e).__name__,
                    component=endpoint
                ).inc()

                raise

            finally:
                # Record latency
                duration = time() - start_time
                application_latency_seconds.labels(
                    endpoint=endpoint
                ).observe(duration)

                # Decrement in-progress gauge
                application_requests_in_progress.labels(endpoint=endpoint).dec()

        return wrapper
    return decorator


def track_llm_metrics(operation: str, model: str = "gpt-4"):
    """
    Decorator to track LLM-specific metrics.

    Usage:
        @track_llm_metrics("fact_extraction", "gpt-4")
        def extract_facts(cv_text):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success
                llm_api_calls_total.labels(
                    operation=operation,
                    model=model,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                # Record failure
                llm_api_calls_total.labels(
                    operation=operation,
                    model=model,
                    status='failure'
                ).inc()

                raise

            finally:
                # Record latency
                duration = time() - start_time
                llm_latency_seconds.labels(
                    operation=operation,
                    model=model
                ).observe(duration)

        return wrapper
    return decorator


# =============================================================================
# METRIC RECORDING FUNCTIONS
# =============================================================================

def record_hallucination_metrics(audit_report):
    """
    Record hallucination-related metrics from audit report.

    Args:
        audit_report: AuditReport from auditor
    """
    # Record hallucination rate
    hallucination_rate.observe(audit_report.hallucination_rate)

    # Record confidence score
    confidence_score.observe(audit_report.overall_confidence)

    # Record flagged if applicable
    if audit_report.flagged:
        audit_flagged_total.labels(reason='high_hallucination').inc()

    # Record claim counts
    claims_total.labels(supported='true').inc(audit_report.supported_claims)
    claims_total.labels(supported='false').inc(audit_report.unsupported_claims)


def record_fact_extraction_metrics(facts_count: int):
    """Record fact extraction metrics."""
    facts_extracted.observe(facts_count)


def record_llm_usage(
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    model: str = "gpt-4"
):
    """
    Record LLM token usage and cost.

    Args:
        operation: Operation name (fact_extraction, generation, etc.)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Cost in USD
        model: Model name
    """
    llm_tokens_total.labels(
        operation=operation,
        token_type='input'
    ).inc(input_tokens)

    llm_tokens_total.labels(
        operation=operation,
        token_type='output'
    ).inc(output_tokens)

    llm_cost_dollars.labels(
        operation=operation,
        model=model
    ).inc(cost)


def record_validation_failure(validation_type: str):
    """
    Record a validation failure.

    Args:
        validation_type: Type of validation (input, schema, business_logic)
    """
    validation_failures_total.labels(
        validation_type=validation_type
    ).inc()


def record_error(error_type: str, component: str):
    """
    Record an application error.

    Args:
        error_type: Type/class of error
        component: Component where error occurred
    """
    application_errors_total.labels(
        error_type=error_type,
        component=component
    ).inc()


# =============================================================================
# METRICS ENDPOINT
# =============================================================================

def get_metrics() -> tuple[bytes, str]:
    """
    Get Prometheus metrics in text format.

    Returns:
        Tuple of (metrics_bytes, content_type)

    Usage in FastAPI:
        @app.get("/metrics")
        def metrics():
            data, content_type = get_metrics()
            return Response(content=data, media_type=content_type)
    """
    return generate_latest(), CONTENT_TYPE_LATEST


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

"""
Example: Instrumenting a cover letter generation endpoint

from backend.app.utils.prometheus_metrics import (
    track_request_metrics,
    record_hallucination_metrics,
    record_fact_extraction_metrics,
    record_llm_usage
)

@app.post("/api/generate")
@track_request_metrics("generate_cover_letter")
async def generate_cover_letter_endpoint(cv: str, job: str):
    # Extract facts
    facts = fact_extractor.extract(cv)
    record_fact_extraction_metrics(len(facts.facts))

    # Generate cover letter
    cover_letter = generate_cover_letter(facts, job)

    # Audit
    audit = auditor.audit(cover_letter, facts)
    record_hallucination_metrics(audit)

    # Record LLM usage (if available from LLM service)
    record_llm_usage(
        operation="total_pipeline",
        input_tokens=500,
        output_tokens=300,
        cost=0.024,
        model="gpt-4"
    )

    return {
        "cover_letter": cover_letter,
        "audit": audit.model_dump()
    }


# Metrics endpoint
@app.get("/metrics")
def prometheus_metrics():
    data, content_type = get_metrics()
    from fastapi import Response
    return Response(content=data, media_type=content_type)
"""
