"""RegIntel version-compare engine.

Provides:
- sentence-level diff between two versions of a regulatory document
- classification of changed sentences as OBLIGATION / PERMISSIVE / INFO
- text extraction from uploaded .txt/.pdf files (pypdf)
- a tiny valid PDF builder used by the tests and available as a utility
"""

from __future__ import annotations

import difflib
import io
import re
from typing import Dict, List, Tuple

from pypdf import PdfReader

OBLIGATION_RE = re.compile(
    r"\b(shall|must|required|require|penalt\w*|mandatory|liable|prohibited|"
    r"obligat\w*|compliance|non-compliance)\b", re.IGNORECASE)
PERMISSIVE_RE = re.compile(
    r"\b(may|optional|permitted|entitled|discretion|voluntar\w*)\b", re.IGNORECASE)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")

CHANGE_TYPES = ("Added", "Removed", "Modified")


def classify(sentence: str) -> str:
    """Classify a sentence as OBLIGATION, PERMISSIVE or INFO."""
    if OBLIGATION_RE.search(sentence):
        return "OBLIGATION"
    if PERMISSIVE_RE.search(sentence):
        return "PERMISSIVE"
    return "INFO"


def split_sentences(text: str) -> List[str]:
    """Sentence split tuned for regulatory prose: headings become their own items."""
    out: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if re.match(r"^(\d+(\.\d+)*\.?\s+\S)", para) and len(para) < 90:
            out.append(para)  # section heading line
        else:
            out.extend(s for s in SENT_SPLIT.split(para) if s.strip())
    return out


def compare_texts(text_a: str, text_b: str) -> Tuple[List[Dict], Dict[str, int]]:
    """Alias for compare_versions (kept as the primary public entry point)."""
    return compare_versions(text_a, text_b)


def compare_versions(text_a: str, text_b: str) -> Tuple[List[Dict], Dict[str, int]]:
    """Public diff API with a clean replace implementation."""
    sent_a = split_sentences(text_a)
    sent_b = split_sentences(text_b)
    matcher = difflib.SequenceMatcher(a=sent_a, b=sent_b, autojunk=False)
    rows: List[Dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        olds = sent_a[i1:i2]
        news = sent_b[j1:j2]
        if tag == "replace":
            used_new: set = set()
            for old in olds:
                best_j, best_ratio = -1, 0.0
                for j, new in enumerate(news):
                    if j in used_new:
                        continue
                    ratio = difflib.SequenceMatcher(a=old, b=new).ratio()
                    if ratio > best_ratio:
                        best_j, best_ratio = j, ratio
                if best_j >= 0 and best_ratio >= 0.5:
                    used_new.add(best_j)
                    new = news[best_j]
                    rows.append({"type": "Modified", "classification": classify(new),
                                 "old": old, "new": new, "similarity": round(best_ratio, 3)})
                else:
                    rows.append({"type": "Removed", "classification": classify(old),
                                 "old": old, "new": "", "similarity": 0.0})
            for j, new in enumerate(news):
                if j not in used_new:
                    rows.append({"type": "Added", "classification": classify(new),
                                 "old": "", "new": new, "similarity": 0.0})
        elif tag == "delete":
            for old in olds:
                rows.append({"type": "Removed", "classification": classify(old),
                             "old": old, "new": "", "similarity": 0.0})
        elif tag == "insert":
            for new in news:
                rows.append({"type": "Added", "classification": classify(new),
                             "old": "", "new": new, "similarity": 0.0})

    counts = {t: 0 for t in CHANGE_TYPES}
    for c in ("OBLIGATION", "PERMISSIVE", "INFO"):
        counts[c] = 0
    for r in rows:
        counts[r["type"]] += 1
        counts[r["classification"]] += 1
    counts["TOTAL"] = len(rows)
    return rows, counts


def summary_counts_md(counts: Dict[str, int]) -> str:
    """Markdown summary line(s) for the Compare tab."""
    return (
        f"**Changes:** {counts['TOTAL']} total | "
        f"Added: {counts['Added']} | Modified: {counts['Modified']} | "
        f"Removed: {counts['Removed']}  \n"
        f"**Classes:** OBLIGATION: {counts['OBLIGATION']} | "
        f"PERMISSIVE: {counts['PERMISSIVE']} | INFO: {counts['INFO']}"
    )


# ---------------------------------------------------------------------------
# File input handling
# ---------------------------------------------------------------------------

def read_upload(path: str) -> str:
    """Read an uploaded .txt or .pdf file and return its text content."""
    if path.lower().endswith(".pdf"):
        return extract_pdf_text(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def extract_pdf_text(path_or_bytes) -> str:
    """Extract text from a PDF file path or bytes using pypdf."""
    reader = PdfReader(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray))
                       else path_or_bytes)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_simple_pdf(lines: List[str]) -> bytes:
    """Build a tiny but valid single-page PDF (Helvetica 12pt) with the given lines.

    Used to generate PDF fixtures without external dependencies.
    """
    content_lines = "\n".join(f"({_pdf_escape(line)}) Tj T*" for line in lines)
    content = f"BT /F1 12 Tf 72 720 Td 16 TL\n{content_lines}\nET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode("latin-1")
    return bytes(out)
