from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_fact_extract_stub():
    r = client.post("/fact-extract", json={"sections": {"skills": "Python"}})
    assert r.status_code == 200
    data = r.json()
    assert data["raw_sections"]["skills"] == "Python"


def test_jd_analyze_mocked(monkeypatch):
    # Patch get_llm_service used by llm router module
    import app.api.routes.llm as llm_routes

    fake = MagicMock()
    fake.generate_response.return_value = {
        "content": '{"requirements":["Python","SQL"],"keywords":["Python","SQL"]}'
    }

    monkeypatch.setattr(llm_routes, "get_llm_service", lambda: fake)

    r = client.post("/jd-analyze", json={"job_description": "Need Python and SQL."})
    assert r.status_code == 200
    data = r.json()
    assert "Python" in data["requirements"]


def test_cover_letter_mocked(monkeypatch):
    import app.api.routes.llm as llm_routes

    fake = MagicMock()
    fake.generate_response.return_value = {"content": '{"cover_letter":"Hello "}'}
    # too short -> will trigger fallback; that's fine

    monkeypatch.setattr(llm_routes, "get_llm_service", lambda: fake)

    r = client.post(
        "/cover-letter",
        json={"facts": {"skills": ["Python"]}, "job": {"requirements": ["Python"]}, "tone": "professional"},
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["cover_letter"], str)
    assert len(data["cover_letter"]) > 0
