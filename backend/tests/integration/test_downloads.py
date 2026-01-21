from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.llm_client import get_llm_client
from pathlib import Path

client = TestClient(app)


class FakeLLMClient:
    async def extract_facts(self, cv_sections):
        return {"experiences": [], "education": [], "skills": [], "raw_sections": cv_sections}

    async def analyze_jd(self, job_description: str):
        return {"requirements": [], "keywords": [], "raw": job_description}

    async def generate_cover_letter(self, facts, jd, tone: str):
        return {"cover_letter": "TEST COVER LETTER", "tone": tone}


def test_generate_then_downloads():
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient()
    try:
        fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sample.docx"
        with open(fixture_path, "rb") as f:
            files = {
                "file": (
                    "sample.docx",
                    f.read(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            }

        data = {
            "job_description": "We are looking for a Python engineer with FastAPI experience and strong testing practices.",
            "tone": "professional",
        }

        r = client.post("/applications/generate", files=files, data=data)
        assert r.status_code == 200, r.text
        request_id = r.json()["request_id"]

        r = client.get(f"/applications/{request_id}/download/cover-letter.docx")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(r.content) > 500

        r = client.get(f"/applications/{request_id}/download/cover-letter.pdf")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content[:4] == b"%PDF"

        r = client.get(f"/applications/{request_id}/download/enhanced-cv.docx")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(r.content) > 500

        r = client.get(f"/applications/{request_id}/download/enhanced-cv.pdf")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content[:4] == b"%PDF"
    finally:
        app.dependency_overrides.clear()
