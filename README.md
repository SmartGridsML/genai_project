# genai_project

# Project Structure

```
cv-application-helper/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry
│   │   ├── config.py                  # Settings (Pydantic BaseSettings)
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── health.py          # /health, /metrics
│   │   │   │   ├── applications.py    # Main endpoint
│   │   │   │   └── jobs.py            # Job tracking endpoints
│   │   │   └── dependencies.py        # Dependency injection
│   │   │
│   │   ├── core/
│   │   │   ├── document_parser.py     # PDF/DOCX → structured data
│   │   │   ├── fact_extractor.py      # CV → JSON facts (LLM)
│   │   │   ├── jd_analyzer.py         # Job Description → requirements
│   │   │   ├── cover_letter_gen.py    # Grounded generation
│   │   │   ├── cv_enhancer.py         # Bullet point improvements
│   │   │   └── auditor.py             # Hallucination detection
│   │   │
│   │   ├── models/
│   │   │   ├── schemas.py             # Pydantic models (API contracts)
│   │   │   └── prompts.py             # All LLM prompts (versioned)
│   │   │
│   │   ├── services/
│   │   │   ├── llm_service.py         # LLM abstraction layer
│   │   │   ├── vector_store.py        # Optional: RAG for CV chunks
│   │   │   ├── document_service.py    # Generate output docs
│   │   │   └── cache_service.py       # Redis caching
│   │   │
│   │   └── utils/
│   │       ├── logging.py             # Structured logging
│   │       ├── metrics.py             # Prometheus metrics
│   │       └── validators.py          # Input validation
│   │
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── test_prompts.py            # Prompt regression tests
│   │
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pyproject.toml
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CVUploader.tsx
│   │   │   ├── JobDescInput.tsx
│   │   │   ├── ResultsViewer.tsx      # Accept/reject UI
│   │   │   └── DiffViewer.tsx         # Show CV changes
│   │   ├── pages/
│   │   └── api/
│   ├── package.json
│   └── Dockerfile
│
├── mlops/
│   ├── monitoring/
│   │   ├── prometheus.yml
│   │   └── grafana-dashboards/
│   ├── experiments/
│   │   └── mlflow_tracking.py         # Track prompt versions
│   └── eval/
│       ├── evaluation_suite.py        # Automated testing
│       └── test_cases.json            # Golden examples
│
├── infra/
│   ├── terraform/                     # AWS/GCP infrastructure
│   │   ├── main.tf
│   │   ├── ecs.tf                     # Container orchestration
│   │   └── api_gateway.tf
│   ├── docker-compose.yml             # Local dev
│   └── kubernetes/                    # Production K8s manifests
│       ├── deployment.yaml
│       └── service.yaml
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Tests + linting
│       ├── cd.yml                     # Deploy pipeline
│       └── prompt-regression.yml      # Test prompt changes
│
└── docs/
    ├── API.md                         # OpenAPI docs
    ├── ARCHITECTURE.md
    └── EVALUATION.md                  # How you measure quality

