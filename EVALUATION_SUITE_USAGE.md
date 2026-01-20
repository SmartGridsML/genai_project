# Evaluation Suite - Quick Start Guide

## What Was Built

A comprehensive evaluation system for the cover letter generation pipeline consisting of:

1. **Golden Test Dataset** (`backend/mlops/eval/test_cases.json`)
   - 8 test cases covering happy path + 7 edge cases
   - Includes expected outputs for regression detection

2. **Automated Evaluation Pipeline** (`backend/mlops/eval/evaluation_suite.py`)
   - Runs full pipeline on each test case
   - Tracks quality, performance, and cost metrics
   - Generates detailed reports
   - Logs to MLflow

3. **Prometheus Metrics** (`backend/app/utils/prometheus_metrics.py`)
   - 15+ production-ready metrics
   - Auto-instrumentation via decorators
   - Dashboard and alert-ready

## Quick Start

### 1. View Available Test Cases

```bash
# Demo without running actual evaluation
python demo_evaluation.py
```

This shows all 8 test cases without requiring API keys.

### 2. Run Full Evaluation (Requires API Keys)

```bash
# Set up environment
export OPENAI_API_KEY="your-key-here"

# Run all test cases
python -m backend.mlops.eval.evaluation_suite

# Run specific test cases
python -m backend.mlops.eval.evaluation_suite \
    --test-ids happy_path_senior_engineer sparse_cv_junior
```

### 3. View Results

Evaluation generates:
- **Console Output**: Real-time progress and summary
- **JSON Report**: Detailed results in `backend/mlops/eval/reports/`
- **MLflow Logs**: Tracked in MLflow for trend analysis

## Understanding Test Cases

Each test case includes:

```json
{
  "id": "unique_identifier",
  "name": "Human-readable name",
  "description": "What this tests",
  "cv_text": "Full CV content...",
  "job_description": "Full JD content...",
  "expected_hallucination_rate": 0.05,
  "expected_facts_count": 10,
  "expected_claims_count": 15,
  "notes": "Why this test exists"
}
```

### Test Coverage

1. **Happy Path**: Well-qualified candidate, strong match
2. **Sparse CV**: Recent graduate with minimal experience
3. **Long JD**: 20+ requirements, candidate meets 60%
4. **Skill Mismatch**: Career changer (mechanical → software)
5. **Overqualified**: PhD applying to junior role
6. **Ambiguous Data**: Employment gaps, unclear dates
7. **Quantitative**: Specific numbers (test accuracy)
8. **International**: Special characters, accents, non-Latin scripts

## Metrics Tracked

### Quality Metrics
- **Hallucination Rate**: % of unsupported claims (target: <5%)
- **Confidence Score**: Average confidence (target: >0.85)
- **Support Ratio**: % of supported claims (target: >95%)

### Performance Metrics
- **P50 Latency**: Median response time
- **P95 Latency**: 95th percentile (target: <30s)
- **P99 Latency**: 99th percentile
- **Total Pipeline Time**: End-to-end duration

### Cost Metrics
- **Token Usage**: Total tokens consumed
- **Estimated Cost**: API cost per request (target: <$0.10)
- **Cost per Component**: Breakdown by pipeline stage

## Interpreting Results

### Console Output

```
================================================================================
EVALUATION COMPLETE
================================================================================
Success Rate: 100.0%
Avg Hallucination Rate: 3.4%
Avg Latency: 9247ms
Total Cost: $0.19
```

### JSON Report

```json
{
  "aggregate_metrics": {
    "avg_hallucination_rate": 0.034,
    "p95_latency_ms": 12500,
    "total_cost_usd": 0.19
  },
  "summary": {
    "passed_quality_threshold": true,
    "passed_latency_threshold": true,
    "overall_pass": true
  }
}
```

### Quality Thresholds

- ✅ **Excellent**: Hallucination rate <5%
- ⚠️ **Acceptable**: Hallucination rate 5-10%
- ❌ **Poor**: Hallucination rate >10%

## Using Prometheus Metrics

### In Your FastAPI App

```python
from app.utils.prometheus_metrics import (
    track_request_metrics,
    record_hallucination_metrics,
    get_metrics
)

# Add metrics endpoint
@app.get("/metrics")
def metrics():
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)

# Instrument endpoints
@app.post("/api/generate")
@track_request_metrics("generate_cover_letter")
async def generate(cv: str, job: str):
    # Your code here
    audit = auditor.audit(result.cover_letter, result.facts)
    record_hallucination_metrics(audit)
    return result
```

### Available Metrics

- `application_requests_total`: Total requests
- `application_latency_seconds`: Request duration histogram
- `hallucination_rate`: Hallucination rate distribution
- `llm_cost_dollars`: API costs
- `llm_tokens_total`: Token usage
- And 10+ more...

## Adding New Test Cases

1. Edit `backend/mlops/eval/test_cases.json`
2. Add your test case following the schema
3. Re-run evaluation

Example:

```json
{
  "id": "my_new_test",
  "name": "Test Description",
  "cv_text": "CV content...",
  "job_description": "JD content...",
  "expected_hallucination_rate": 0.05,
  "expected_facts_count": 12,
  "expected_claims_count": 15,
  "notes": "Why this test matters"
}
```

## Regression Testing

### Before Making Changes

```bash
# Establish baseline
python -m backend.mlops.eval.evaluation_suite > baseline.txt

# Note metrics:
# - Avg Hallucination Rate: 3.4%
# - P95 Latency: 12.5s
# - Total Cost: $0.19
```

### After Making Changes

```bash
# Re-run evaluation
python -m backend.mlops.eval.evaluation_suite > after_change.txt

# Compare:
# - Hallucination: 3.4% → 2.1% ✓ (improved!)
# - Latency: 12.5s → 15.2s ✗ (slower)
# - Cost: $0.19 → $0.27 ✗ (more expensive)

# Decision: Is quality improvement worth latency/cost increase?
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Run Evaluation Suite

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run Evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m backend.mlops.eval.evaluation_suite
      - name: Check Thresholds
        run: |
          # Fail if metrics exceed thresholds
          python scripts/check_regression.py
```

## Troubleshooting

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'app'`

**Solution**: All imports have been updated to use `backend.app.*`. Ensure you're running from the project root.

### MLflow Warnings

**Issue**: `Malformed experiment '0'. Detailed error Yaml file './mlruns/0/meta.yaml' does not exist.`

**Solution**: Clean up corrupted MLflow directories:
```bash
rm -rf ./mlruns
mkdir -p ./mlruns
```

### Missing API Key

**Issue**: Evaluation fails with authentication error

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Next Steps

1. **Run Evaluation**: Test the current system
2. **Add Test Cases**: Expand to 20-50 cases over time
3. **Set Up CI/CD**: Run automatically on every PR
4. **Monitor Production**: Sample real traffic
5. **Iterate**: Use metrics to improve quality

## Learning Resources

- **Evaluation Guide**: `10x/evaluation-strategies-guide.md` (10,000 words)
- **Metrics Guide**: `10x/production-metrics-monitoring-guide.md` (12,000 words)
- **Implementation**: Study `backend/mlops/eval/evaluation_suite.py`
- **Test Cases**: Review `backend/mlops/eval/test_cases.json`

## Summary

The evaluation suite provides:

✅ Automated quality assurance
✅ Quantitative metrics (quality, performance, cost)
✅ Regression detection
✅ Production monitoring
✅ Data-driven decision making

**You now have a production-ready evaluation system!** 🚀
