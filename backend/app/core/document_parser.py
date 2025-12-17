from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import io

from docx import Document
from pypdf import PdfReader


SECTION_ALIASES: Dict[str, List[str]] = {
    "experience": ["experience", "work experience", "employment", "professional experience"],
    "education": ["education", "academic", "academics"],
    "skills": ["skills", "technical skills", "core skills", "competencies", "technologies"],
    "projects": ["projects", "personal projects", "selected projects"],
    "summary": ["summary", "profile", "about", "professional summary"],
}

DEFAULT_SECTION_ORDER = ["summary", "experience", "projects", "education", "skills"]


@dataclass
class ParsedCV:
    raw_text: str
    sections: Dict[str, str]          # canonical section_name -> section text
    detected_headings: List[str]      # headings found in the text


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive spaces
    text = re.sub(r"[ \t]+", " ", text)
    # collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return _normalize("\n".join(pages))


def _extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _normalize("\n".join(paras))


def _build_heading_regex() -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    for canonical, aliases in SECTION_ALIASES.items():
        for a in aliases:
            # Match headings that appear on their own line, optionally with ":".
            # Case-insensitive; tolerate leading/trailing spaces.
            pat = re.compile(rf"(?im)^\s*{re.escape(a)}\s*:?\s*$")
            patterns.append((canonical, pat))
    return patterns


HEADING_PATTERNS = _build_heading_regex()


def split_into_sections(text: str) -> ParsedCV:
    """
    Heuristic section splitter:
    - Finds lines that look like common section headings
    - Splits text between headings
    """
    lines = text.split("\n")
    # Identify heading line indices
    heading_hits: List[Tuple[int, str, str]] = []  # (idx, canonical, original_line)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for canonical, pat in HEADING_PATTERNS:
            if pat.match(stripped):
                heading_hits.append((i, canonical, stripped))
                break

    detected = [h[2] for h in heading_hits]

    # If no headings found, return raw text as a single "raw" section
    if not heading_hits:
        return ParsedCV(raw_text=text, sections={"raw": text}, detected_headings=[])

    # Merge duplicates: keep first occurrence; later duplicates treated as normal text boundaries
    # We'll still split at every heading; the "canonical" name can repeat and will be appended.
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_ALIASES.keys()}
    sections.setdefault("raw", [])

    # Create boundaries
    boundaries = [(idx, canonical) for idx, canonical, _ in heading_hits]
    boundaries.append((len(lines), "__end__"))

    for b in range(len(boundaries) - 1):
        start_idx, canonical = boundaries[b]
        end_idx, _ = boundaries[b + 1]

        # Capture everything after heading line until next heading
        chunk_lines = lines[start_idx + 1 : end_idx]
        chunk = _normalize("\n".join(chunk_lines))
        if not chunk:
            continue

        if canonical not in sections:
            sections["raw"].append(chunk)
        else:
            sections[canonical].append(chunk)

    # Flatten
    flat: Dict[str, str] = {}
    for k, chunks in sections.items():
        joined = _normalize("\n\n".join(chunks)) if chunks else ""
        if joined:
            flat[k] = joined

    # Provide a stable order by adding missing known sections as empty if needed (optional)
    return ParsedCV(raw_text=text, sections=flat, detected_headings=detected)


def parse_cv(file_bytes: bytes, filename: str) -> ParsedCV:
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _extract_pdf_text(file_bytes)
    elif name.endswith(".docx"):
        text = _extract_docx_text(file_bytes)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .docx are allowed.")

    if not text or len(text) < 50:
        # 50 chars is a sanity threshold; can tune later
        raise ValueError("Could not extract enough text from the document.")

    return split_into_sections(text)
