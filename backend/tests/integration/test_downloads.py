import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api.routes import applications as applications_routes
from backend.app.main import app


class FakeLLMClient:
    async def extract_facts(self, cv_sections):
        return {"facts": [{"category": "experience", "fact": "Python engineer", "confidence": 0.9}]}

    async def analyze_jd(self, job_description: str):
        return {
            "summary": "Python role",
            "required_skills": ["Python"],
            "experience_level": "mid",
            "remote_policy": "hybrid",
        }

    async def generate_cover_letter(self, *, facts, jd, tone: str = "professional"):
        return {
            "cover_letter": (
                "I am excited to apply for this Python engineering role. "
                "My experience with Python and FastAPI directly aligns with your requirements."
            )
        }

    async def generate_text(
        self,
        *,
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=1000,
        json_mode=False,
    ):
        if '"claims"' in system_prompt and '"verifications"' not in system_prompt:
            return {"content": '{"claims": ["I have Python experience."]}', "provider": "fake"}
        return {
            "content": (
                '{"verifications":[{"claim":"I have Python experience.","supported":true,'
                '"source":"CV fact: Python engineer","confidence":0.95,"reasoning":"Exact match"}]}'
            ),
            "provider": "fake",
        }


def _wait_for_done(client: TestClient, request_id: str, access_token: str, timeout_seconds: float = 10.0) -> dict:
    deadline = time.time() + timeout_seconds
    headers = {"Authorization": f"Bearer {access_token}"}

    while time.time() < deadline:
        res = client.get(f"/v1/applications/{request_id}/results", headers=headers)
        assert res.status_code == 200, res.text
        payload = res.json()
        if payload.get("status") in {"done", "failed"}:
            return payload
        time.sleep(0.1)

    raise AssertionError("Timed out waiting for generation to complete")


def test_generate_then_downloads(monkeypatch):
    monkeypatch.setattr(applications_routes, "get_llm_client", lambda: FakeLLMClient())
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sample.docx"

    with TestClient(app) as client:
        with open(fixture_path, "rb") as f:
            parse_r = client.post(
                "/v1/applications/parse",
                files={
                    "file": (
                        "sample.docx",
                        f.read(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
        assert parse_r.status_code == 200, parse_r.text
        parsed_text = parse_r.json()["parsed_text"]
        assert parsed_text

        gen_r = client.post(
            "/v1/applications/generate",
            json={
                "cv_text": parsed_text,
                "job_description": (
                    "We are looking for a Python engineer with FastAPI experience and strong testing practices."
                ),
                "tone": "professional",
            },
        )
        assert gen_r.status_code == 200, gen_r.text
        request_id = gen_r.json()["request_id"]
        access_token = gen_r.json()["access_token"]
        assert gen_r.json()["status"] == "processing"

        final = _wait_for_done(client, request_id, access_token)
        assert final["status"] == "done", final

        headers = {"Authorization": f"Bearer {access_token}"}

        r = client.get(f"/v1/applications/{request_id}/download/cover-letter.docx", headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(r.content) > 500

        r = client.get(f"/v1/applications/{request_id}/download/cover-letter.pdf", headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content[:4] == b"%PDF"

        r = client.get(f"/v1/applications/{request_id}/download/enhanced-cv.docx", headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(r.content) > 500

        r = client.get(f"/v1/applications/{request_id}/download/enhanced-cv.pdf", headers=headers)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content[:4] == b"%PDF"
