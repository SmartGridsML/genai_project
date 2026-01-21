from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_generate_then_downloads():
    # Use a DOCX because it's easy to craft bytes? But simplest is to ship a fixture file.
    # If you already have a test fixture PDF/DOCX, use it.
    # For now, we use a minimal DOCX header-less bytes won't parse.
    # So: use an existing fixture in your repo if available.

    # --- Replace with your real fixture path if you have one ---
    # Example assumes backend/tests/fixtures/sample.docx exists.
    fixture_path = "backend/tests/fixtures/sample.docx"
    with open(fixture_path, "rb") as f:
        files = {"file": ("sample.docx", f.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}

    data = {
        "job_description": "We are looking for a Python engineer with FastAPI experience and strong testing practices.",
        "tone": "professional",
    }

    r = client.post("/applications/generate", files=files, data=data)
    assert r.status_code == 200, r.text
    request_id = r.json()["request_id"]

    # cover letter docx
    r = client.get(f"/applications/{request_id}/download/cover-letter.docx")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert len(r.content) > 500  # docx is zipped xml, should be > tiny

    # cover letter pdf
    r = client.get(f"/applications/{request_id}/download/cover-letter.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content[:4] == b"%PDF"

    # enhanced cv docx
    r = client.get(f"/applications/{request_id}/download/enhanced-cv.docx")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert len(r.content) > 500

    # enhanced cv pdf
    r = client.get(f"/applications/{request_id}/download/enhanced-cv.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content[:4] == b"%PDF"
