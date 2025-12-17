import pytest
from app.core.document_parser import split_into_sections


def test_split_into_sections_detects_basic_headings():
    text = """
SUMMARY
Data scientist with experience...

EXPERIENCE
- Built models

EDUCATION
MSc Something

SKILLS
Python, SQL
""".strip()

    parsed = split_into_sections(text)
    assert "experience" in parsed.sections
    assert "education" in parsed.sections
    assert "skills" in parsed.sections
    assert "Built models" in parsed.sections["experience"]
