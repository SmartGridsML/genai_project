import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.app.core.auditor import Auditor, AuditorError
from backend.app.models.schemas import ExtractedFacts, KeyFact


@pytest.fixture
def fact_table() -> ExtractedFacts:
    return ExtractedFacts(
        facts=[
            KeyFact(category="experience", fact="Python engineer at Acme", confidence=0.95),
            KeyFact(category="skills", fact="FastAPI and Docker", confidence=0.9),
        ]
    )


def test_audit_rejects_empty_inputs(fact_table: ExtractedFacts) -> None:
    llm = AsyncMock()
    auditor = Auditor(llm_client=llm)

    with pytest.raises(AuditorError):
        asyncio.run(auditor.audit("", fact_table))

    with pytest.raises(AuditorError):
        asyncio.run(auditor.audit("Valid letter", ExtractedFacts(facts=[])))


def test_audit_happy_path_batch_verification(fact_table: ExtractedFacts) -> None:
    llm = AsyncMock()
    llm.generate_text = AsyncMock(
        side_effect=[
            {"content": '{"claims": ["Python engineer at Acme"]}', "provider": "fake"},
            {
                "content": (
                    '{"verifications":[{"claim":"Python engineer at Acme","supported":true,'
                    '"source":"CV fact: Python engineer at Acme","confidence":1.0,'
                    '"reasoning":"Exact match"}]}'
                ),
                "provider": "fake",
            },
        ]
    )

    auditor = Auditor(llm_client=llm)
    report = asyncio.run(
        auditor.audit(
            cover_letter="I worked as a Python engineer at Acme.",
            fact_table=fact_table,
            request_id="req-1",
        )
    )

    assert report.total_claims == 1
    assert report.supported_claims == 1
    assert report.unsupported_claims == 0
    assert report.flagged is False
    assert llm.generate_text.await_count == 2


def test_audit_falls_back_to_sequential_when_batch_parse_fails(fact_table: ExtractedFacts) -> None:
    llm = AsyncMock()
    llm.generate_text = AsyncMock(
        side_effect=[
            {"content": '{"claims": ["Python engineer at Acme", "Knows FastAPI"]}', "provider": "fake"},
            {"content": '{"verifications": "broken"}', "provider": "fake"},
            {
                "content": (
                    '{"supported":true,"source":"CV fact: Python engineer at Acme",'
                    '"confidence":0.95,"reasoning":"Match"}'
                ),
                "provider": "fake",
            },
            {
                "content": (
                    '{"supported":true,"source":"CV fact: FastAPI and Docker",'
                    '"confidence":0.9,"reasoning":"Match"}'
                ),
                "provider": "fake",
            },
        ]
    )

    auditor = Auditor(llm_client=llm)
    report = asyncio.run(
        auditor.audit(
            cover_letter="I worked as a Python engineer at Acme and know FastAPI.",
            fact_table=fact_table,
        )
    )

    assert report.total_claims == 2
    assert report.supported_claims == 2
    assert report.unsupported_claims == 0
    assert llm.generate_text.await_count == 4
