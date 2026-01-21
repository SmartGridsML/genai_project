import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app

client = TestClient(app)


def test_generate_then_downloads():
    settings.llm_base_url = None

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
    assert r.status_code == 202, r.text
    request_id = r.json()["request_id"]

    status_payload = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status_resp = client.get(f"/applications/{request_id}/status")
        assert status_resp.status_code == 200
        status_payload = status_resp.json()
        if status_payload.get("status") == "completed":
            break
        time.sleep(0.2)

    assert status_payload is not None
    assert status_payload.get("status") == "completed"

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
