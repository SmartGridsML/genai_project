# Architecture

## System Design Diagram

```
                   +---------------------+
                   |  Web / API Clients  |
                   +----------+----------+
                              |
                              v
                     +--------+--------+
                     |  Application LB |
                     +--------+--------+
                              |
                              v
                   +----------+----------+
                   |   ECS Fargate       |
                   |  FastAPI backend    |
                   +----+-----------+----+
                        |           |
                        |           +-------------------+
                        |                               |
                        v                               v
             +----------+-----------+        +----------+-----------+
             |  Redis (cache)       |        |  Postgres (db)       |
             +----------------------+        +---------------------+
                        |
                        v
             +----------+-----------+
             |  S3 Document Store   |
             +----------------------+

   LLM Providers:
   +-----------------------+       +-------------------------+
   | OpenAI (primary)      | <---- | Gemini (fallback)       |
   +-----------------------+       +-------------------------+

   Observability:
   +-----------------------+       +-------------------------+
   | Prometheus /metrics   | ----> | Managed Prometheus      |
   +-----------------------+       +-------------------------+
               |                          |
               v                          v
        +------+--------+           +-----+-------+
        | Grafana UI    |           | CloudWatch  |
        +---------------+           +-------------+
```

## Anti-Hallucination Strategy

The system uses a multi-stage, defense-in-depth approach to prevent fabricated claims:

1) Fact extraction with schema validation
- CV text is converted into structured facts using strict JSON-only prompts.
- The response is parsed and validated; invalid JSON is rejected.
- Prompt versions are centralized to keep behavior consistent and auditable.

2) Claim extraction from generated content
- The auditor extracts factual claims from the generated cover letter.
- Claims are normalized to a consistent schema for verification.

3) Claim verification against extracted facts
- Each claim is checked against the fact table.
- Unsupported claims are flagged and counted toward the hallucination rate.

4) Explicit quality gating
- The hallucination rate is computed per request.
- Requests exceeding a configurable threshold are marked as risky.
- Metrics are emitted for alerting and regression tracking.

5) Provider fallback and prompt discipline
- OpenAI is primary, Gemini is fallback.
- JSON mode and explicit instructions reduce free-form drift.

## Latency Budget (p95)

Total target: 30 seconds per request

- Upload + parsing: 2s
- Fact extraction (LLM): 8s
- Cover letter generation (LLM): 8s
- Hallucination audit (LLM): 6s
- DOCX/PDF rendering: 3s
- Storage + network overhead: 3s

## Cost Analysis (per request, estimate)

Assumptions:
- 3 LLM calls: fact extraction, generation, audit
- Total input tokens: 4.5k
- Total output tokens: 2.8k
- Example pricing: $5 / 1M input tokens, $15 / 1M output tokens

Estimated LLM cost:
- Input: 4.5k * $5 / 1,000,000 = $0.0225
- Output: 2.8k * $15 / 1,000,000 = $0.0420
- LLM total: ~$0.0645 per request

Infrastructure (monthly, example scale):
- ECS Fargate (2 tasks, 0.5 vCPU / 1GB): ~$40-80
- ALB + data processing: ~$20-50
- Redis + Postgres (managed): varies by size
- S3 storage + requests: low unless high volume
- Monitoring (CloudWatch + managed services): ~$10-50

This yields a per-request cost dominated by LLM usage at low to moderate traffic. At scale, compute and storage overheads remain secondary to LLM spend.

## Comprehensive Cost Strategy

To ensure long-term sustainability and 10x efficiency, the following strategies will be employed to optimize costs across LLM, Infrastructure, and Operations.

### 1. LLM Cost Optimization (The "Big Rock")
Since LLM costs account for >80% of per-request expenses, this is the primary area for optimization.

*   **Model Tiering**:
    *   **Auditor Extraction Phase**: Switch from GPT-4o to **Gemini Flash** or **GPT-4o-mini**. Extracting claims is a simpler NLP task that cheaper models handle well.
    *   **Fact Extraction Phase**: Use **GPT-3.5-turbo** or **Gemini Flash**. The strict schema validation layer compensates for the slightly lower reasoning capability.
    *   **Generation Phase**: Reserve the premium models (GPT-4o) *only* for the actual writing of the cover letter to ensure high prose quality.
    *   **Projected Savings**: ~60% reduction in token costs.

*   **Aggressive Caching (Redis)**:
    *   **Fact Extraction**: Hash the input CV text. If a user regenerates a cover letter for a different job application using the same CV, return the cached extracted facts immediately.
    *   **TTL**: Set a 24-hour TTL for cached artifacts to balance freshness with savings.

*   **Prompt Compression**:
    *   Minimize system prompt verbosity without losing instruction clarity.
    *   Remove examples from prompts once the model performance is stable (fine-tuning is a longer-term option).

### 2. Infrastructure Optimization
*   **Fargate Spot Instances**:
    *   Migrate non-critical or stateless workloads (like the API tasks) to **Fargate Spot**.
    *   **Savings**: Up to 70% off on-demand Fargate prices.
    *   **Risk Mitigation**: Ensure the Application Load Balancer (ALB) has aggressive health checks and quick drainage.

*   **S3 Lifecycle Policies**:
    *   Generated cover letters and resumes are ephemeral in value.
    *   **Policy**: Move objects to **Glacier Instant Retrieval** after 30 days and **Expire (Delete)** after 90 days.

*   **Database Rightsizing**:
    *   Start with **Aurora Serverless v2** for Postgres to scale compute down to 0.5 ACU during low-traffic periods (nights/weekends).

### 3. Operational Guardrails
*   **Rate Limiting**: Implement strict per-user rate limits (e.g., 10 requests/hour) to prevent accidental or malicious wallet draining.
*   **Budget Alerts**: Set AWS Budgets to alert via SNS/Email when daily spend exceeds a forecasted threshold (e.g., $10/day).