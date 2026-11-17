"""RegIntel core retrieval and answering engine.

Provides:
- document chunking with metadata (title, section, page estimate)
- BM25 retrieval over the bundled corpus (rank_bm25)
- demo-mode deterministic answer assembly with inline citations
- citation guard that flags sentences not supported by retrieved evidence
- optional live-mode LLM synthesis via an OpenAI-compatible chat completions API
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

DEFAULT_CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

DEMO_HEADER = "[DEMO MODE — deterministic summary from retrieved evidence]"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "shall", "may", "can", "which", "who", "whom", "this", "such",
    "not", "no", "any", "all", "also", "been", "being", "into", "under", "than",
    "then", "there", "their", "them", "they", "he", "she", "his", "her", "him",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?\s+\S|SECTION\s|PART\s|ANNEX|APPENDIX)")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in raw if len(t) > 1 and t not in _STOPWORDS]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, keeping heading lines intact."""
    out: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if _HEADING_RE.match(para) and len(para) < 90:
            out.append(para)
        else:
            out.extend(s for s in _SENT_SPLIT.split(para) if s.strip())
    return out


def content_ngrams(text: str, n: int = 4) -> set:
    """Set of lowercase content-word n-grams (n words, punctuation removed)."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]
    if n <= 1:
        return set(words)
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    page: int
    text: str
    words: int = field(init=False)

    def __post_init__(self) -> None:
        self.words = len(self.text.split())

    def citation_label(self, n: int) -> str:
        return f"{n}. **{self.title}** — {self.section}, p. {self.page}"

    def snippet(self, max_chars: int = 420) -> str:
        return self.text if len(self.text) <= max_chars else self.text[:max_chars].rstrip() + "..."


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _HEADING_RE.match(s):
        return True
    if len(s) <= 70 and s.upper() == s and any(c.isalpha() for c in s) and not s.endswith("."):
        return True
    return False


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Split raw document text into (section_heading, section_body) pairs."""
    lines = text.split("\n")
    sections: List[Tuple[str, List[str]]] = [("Preamble", [])]
    for line in lines:
        if _is_heading(line):
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def chunk_doc(text: str, doc_id: str, title: str, max_words: int = 160) -> List[Chunk]:
    """Chunk a document into sections (split further when a section is large).

    The leading "Title:" metadata line is stored as the chunk title and removed
    from the chunk body so it is never quoted in answers.
    """
    body_lines = [ln for ln in text.split("\n")
                  if not ln.strip().lower().startswith("title:")]
    text = "\n".join(body_lines).strip()
    chunks: List[Chunk] = []
    words_before = 0
    for heading, body in _split_sections(text):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        groups: List[List[str]] = []
        current: List[str] = []
        count = 0
        for para in paragraphs:
            pw = len(para.split())
            if current and count + pw > max_words:
                groups.append(current)
                current, count = [], 0
            current.append(para)
            count += pw
        if current:
            groups.append(current)
        for gi, group in enumerate(groups, start=1):
            label = heading if len(groups) == 1 else f"{heading} (part {gi})"
            text_chunk = "\n\n".join(group)
            page = words_before // 300 + 1
            chunks.append(Chunk(doc_id=doc_id, title=title, section=label, page=page, text=text_chunk))
            words_before += len(text_chunk.split())
    return chunks


class CorpusIndex:
    """Loads every .txt file in the corpus directory and indexes its chunks."""

    def __init__(self, corpus_dir: str = DEFAULT_CORPUS_DIR) -> None:
        self.corpus_dir = corpus_dir
        self.docs: List[Dict] = []
        self.chunks: List[Chunk] = []
        for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if not lines:
                continue
            raw_title = lines[0]
            title = raw_title[len("Title:"):].strip() if raw_title.lower().startswith("title:") else raw_title
            doc_id = os.path.splitext(os.path.basename(path))[0]
            self.docs.append({
                "doc_id": doc_id,
                "title": title,
                "path": path,
                "words": len(text.split()),
                "sections": sum(1 for ln in lines if _is_heading(ln)),
            })
            self.chunks.extend(chunk_doc(text, doc_id=doc_id, title=title))
        if self.chunks:
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        else:
            self._bm25 = None

    def doc_table(self) -> List[List]:
        """Rows for the Corpus Browser table: id, title, words, sections, chunks."""
        chunk_counts: Dict[str, int] = {}
        for c in self.chunks:
            chunk_counts[c.doc_id] = chunk_counts.get(c.doc_id, 0) + 1
        return [[d["doc_id"], d["title"], d["words"], d["sections"], chunk_counts.get(d["doc_id"], 0)]
                for d in self.docs]

    def search(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top-k chunks for a query with their BM25 scores."""
        if not self._bm25:
            return []
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[Tuple[Chunk, float]] = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= k:
                break
            hits.append((self.chunks[i], float(scores[i])))
        return hits

    def get_doc_text(self, doc_id: str) -> str:
        for d in self.docs:
            if d["doc_id"] == doc_id:
                with open(d["path"], "r", encoding="utf-8") as fh:
                    return fh.read()
        return ""


_INDEX: Optional[CorpusIndex] = None


def get_index() -> CorpusIndex:
    """Lazily built process-wide corpus index."""
    global _INDEX
    if _INDEX is None:
        _INDEX = CorpusIndex()
    return _INDEX


def _best_sentence(chunk: Chunk, q_tokens: Sequence[str]) -> str:
    """Most query-relevant sentence inside a chunk (falls back to first body sentence).

    Metadata lines (Title:/Version:) and heading lines are never quoted.
    """
    candidates = []
    for s in split_sentences(chunk.text):
        s = s.strip()
        if not s or len(s.split()) <= 6:
            continue
        if s.lower().startswith(("title:", "version:")):
            continue
        if _HEADING_RE.match(s) and len(s) < 90:
            continue
        candidates.append(s)
    if not candidates:
        candidates = [ln.strip() for ln in chunk.text.split("\n")
                      if len(ln.split()) > 8
                      and not ln.strip().lower().startswith(("title:", "version:"))]
    if not candidates:
        body = "\n".join(ln for ln in chunk.text.split("\n")
                         if not ln.strip().lower().startswith(("title:", "version:"))).strip()
        return body.split("\n")[0] if body else chunk.text.split("\n")[0]
    q_set = set(q_tokens)
    best, best_score = candidates[0], -1
    for s in candidates:
        overlap = len(q_set & set(tokenize(s)))
        score = overlap / (1 + abs(len(q_set) - len(set(tokenize(s)))) / 20.0)
        if score > best_score:
            best, best_score = s, score
    return best


def demo_answer(query: str, hits: List[Tuple[Chunk, float]]) -> str:
    """Deterministic summary assembled strictly from the retrieved chunks."""
    q_tokens = tokenize(query)
    lines = [DEMO_HEADER, ""]
    for i, (chunk, _score) in enumerate(hits, start=1):
        quote = _best_sentence(chunk, q_tokens)
        lines.append(f"{i}. Under **{chunk.title}** ({chunk.section}, p. {chunk.page}): "
                     f"\"{quote}\" [{i}]")
    lines.append("")
    lines.append(f"Retrieved {len(hits)} evidence chunk(s) from the bundled corpus via BM25.")
    return "\n".join(lines)


def citation_guard(answer: str, evidence_texts: Sequence[str], n: int = 4) -> List[str]:
    """Flag answer sentences that share no content n-gram with the evidence.

    Sentences shorter than 6 content words (connective scaffolding) are ignored.
    Returns the list of flagged sentences; empty list means fully supported.
    """
    evidence_ngrams: set = set()
    for t in evidence_texts:
        evidence_ngrams |= content_ngrams(t, n)
        # also allow shorter shingles so sub-sentence quotes are recognised
        evidence_ngrams |= content_ngrams(t, 2)
    flagged: List[str] = []
    for line in answer.split("\n"):
        s = line.strip()
        if not s or s.startswith("[DEMO") or s.startswith("Retrieved"):
            continue
        # strip citation markers and markdown
        s = re.sub(r"\[\d+\]", "", s)
        s = re.sub(r"[*_>#`]", "", s)
        for sent in split_sentences(s):
            grams = content_ngrams(sent, n)
            small = content_ngrams(sent, 2)
            if len(small) < 6:
                continue  # too short to evaluate as a claim
            if not (grams & evidence_ngrams) and not (small & evidence_ngrams):
                flagged.append(sent.strip())
    return flagged


def guard_report(answer: str, evidence_texts: Sequence[str]) -> str:
    """Human-readable citation guard line for the UI."""
    flagged = citation_guard(answer, evidence_texts)
    if flagged:
        return "Citation guard: FLAGGED — sentences without supporting evidence:\n" + "\n".join(
            f"- {s}" for s in flagged)
    return "Citation guard: All claims supported by the cited evidence."


def evidence_panel(hits: List[Tuple[Chunk, float]]) -> str:
    """Markdown panel showing the chunks cited in the answer."""
    if not hits:
        return "_No evidence retrieved._"
    parts: List[str] = []
    for i, (chunk, score) in enumerate(hits, start=1):
        parts.append(f"### [{i}] {chunk.title}\n\n"
                     f"*Section:* {chunk.section} | *Page:* {chunk.page} | "
                     f"*Doc:* {chunk.doc_id} | *BM25 score:* {score:.2f}\n\n"
                     f"> {chunk.snippet()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Live (LLM) mode
# ---------------------------------------------------------------------------

def _llm_config() -> Tuple[str, str, str]:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return base_url, api_key, model


def call_llm(query: str, hits: List[Tuple[Chunk, float]]) -> str:
    """Call an OpenAI-compatible chat completions endpoint. Raises on failure."""
    import json as _json

    import requests

    base_url, api_key, model = _llm_config()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    context_blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        context_blocks.append(f"[{i}] {chunk.title} — {chunk.section}, p. {chunk.page}\n{chunk.text}")
    system_prompt = (
        "You are RegIntel, a regulatory change assistant. Answer the user's question using ONLY "
        "the numbered context passages provided. Every sentence of your answer MUST end with an "
        "inline citation like [1] or [2] referring to the passage it draws from. Do not invent "
        "facts, sections or numbers that are not present in the context. If the context is "
        "insufficient, say so in a sentence citing the closest passage."
    )
    user_prompt = "Context passages:\n\n" + "\n\n".join(context_blocks) + f"\n\nQuestion: {query}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=_json.dumps(payload),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM returned an empty response")
    return content.strip()


def ask(query: str, mode: str = "demo", index: Optional[CorpusIndex] = None) -> Dict[str, str]:
    """Answer a question in demo or live mode.

    Returns a dict with keys: answer, guard, evidence, mode, notice.
    Live mode falls back to the demo engine on any LLM failure.
    """
    idx = index if index is not None else get_index()
    hits = idx.search(query, k=3)
    if not hits:
        return {
            "answer": "No relevant passages were found in the corpus for this question.",
            "guard": "Citation guard: nothing to verify (no evidence retrieved).",
            "evidence": "",
            "mode": "demo",
            "notice": "",
        }
    evidence_texts = [c.text for c, _ in hits]
    if mode == "live" and os.environ.get("OPENAI_API_KEY"):
        try:
            llm_text = call_llm(query, hits)
            return {
                "answer": llm_text,
                "guard": guard_report(llm_text, evidence_texts),
                "evidence": evidence_panel(hits),
                "mode": "live",
                "notice": "",
            }
        except Exception as exc:  # graceful fallback
            answer = demo_answer(query, hits)
            return {
                "answer": answer,
                "guard": guard_report(answer, evidence_texts),
                "evidence": evidence_panel(hits),
                "mode": "demo",
                "notice": f"LLM call failed ({type(exc).__name__}); fell back to the demo engine.",
            }
    answer = demo_answer(query, hits)
    notice = ""
    if mode == "live":
        notice = "OPENAI_API_KEY is not set; using the offline demo engine."
    return {
        "answer": answer,
        "guard": guard_report(answer, evidence_texts),
        "evidence": evidence_panel(hits),
        "mode": "demo",
        "notice": notice,
    }
