# RegIntel — Citation-Grounded Regulatory Change Intelligence

RegIntel is a retrieval-augmented generation (RAG) system for regulatory research where
**every answer sentence must carry an inline citation** to a retrieved document chunk.
It addresses the core accountability problem of generic chatbots: hallucinated answers
with no traceable evidence.

Three capabilities ship out of the box:

1. **Ask the Corpus** — BM25 retrieval over a bundled corpus of regulatory documents.
   Answers carry inline citations `[1][2][3]`, an evidence panel shows the cited chunks,
   and a **citation guard** reports either "All claims supported" or lists the sentences
   that lack supporting evidence.
2. **Compare Versions** — sentence-level diff between two versions of a circular, with
   every Added / Modified / Removed sentence classified as **OBLIGATION**
   (shall / must / required / penalty / mandatory), **PERMISSIVE** (may / optional /
   permitted) or **INFO**, plus summary counts.
3. **Corpus Browser** — corpus statistics (word counts, sections, chunks) with full-text
   document preview.

## Problem statement

The bundled corpus mirrors a BFSI / regulated-industry compliance desk
(GST rate circulars, RBI KYC master directions, SEBI LODR, FSSAI) — the
same workload a bank or NBFC compliance team faces daily.

Enterprises receive hundreds of regulatory circulars and directives. Generic chatbots
answer questions about them without accountability: answers are unverifiable and often
fabricated. RegIntel enforces grounding mechanically — every claim is checked against
retrieved evidence — and tracks how obligations change between document versions.

## Architecture

```
corpus/*.txt          12 bundled sample regulatory documents (SAMPLE DATA)
engine.py             chunking (title/section/page metadata), BM25 index (rank_bm25),
                      demo answer assembly, citation guard (n-gram overlap),
                      OpenAI-compatible LLM call (requests) + graceful fallback
versioning.py         sentence diff (difflib), obligation classification,
                      pypdf text extraction, minimal PDF builder (test fixture)
app.py                Gradio Blocks UI (3 tabs)
tests/test_core.py    15 unittest cases
smoke_test.py         headless launch + HTTP 200 check on port 7861
```

Retrieval pipeline: document -> section-aware chunking (sections become chunks; large
sections are split at ~160 words) -> BM25Okapi scoring -> top-3 chunks with
`title / section / page` metadata -> answer assembly or LLM synthesis -> citation guard.

The citation guard converts each answer sentence and each evidence chunk into
content-word n-gram sets (4-grams plus 2-grams for sub-sentence quotes). A sentence is
flagged when it shares no n-gram with the cited evidence; short scaffolding sentences
(fewer than 6 content words) are skipped.

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7861. The bundled GST version pair
(`gst_circular_12_2023.txt` vs `gst_circular_12_2023r.txt`) demos the Compare tab
out of the box.

## DEMO vs LIVE mode

- **DEMO mode (default, no network):** used whenever `OPENAI_API_KEY` is not set, or
  when the Demo radio option is selected. The answer is assembled deterministically by
  quoting the most query-relevant sentence of each top-3 retrieved chunk, prefixed with
  `[DEMO MODE — deterministic summary from retrieved evidence]`. Every sentence carries
  a real citation.
- **LIVE mode:** set environment variables and select "Live (LLM)":

  | Variable          | Default                     |
  |-------------------|-----------------------------|
  | `OPENAI_API_KEY`  | (none; required for live)   |
  | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
  | `LLM_MODEL`       | `gpt-4o-mini`               |

  The LLM is prompted to synthesize an answer where every sentence ends with an inline
  citation to a numbered context passage. The response is then passed through the same
  citation guard; unsupported sentences are reported in the UI. On any LLM failure
  (missing key, network error, empty response) the app falls back to the demo engine
  and shows a notice.

## Deploy to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK (any Python 3.10+ runtime).
2. Push these files to the Space repository: `app.py`, `engine.py`, `versioning.py`,
   `corpus/`, `requirements.txt`, `README.md`.
3. In Space settings -> Variables and secrets, add `OPENAI_API_KEY` (secret) and,
   optionally, `OPENAI_BASE_URL` and `LLM_MODEL`.
4. The app auto-detects the key: with a secret it defaults to Live mode; without one it
   stays in offline Demo mode. No changes to the code are needed.

## Tests

```bash
python -m unittest discover -s tests -v
```

15 tests cover: chunker metadata correctness, BM25 planted-query retrieval, citation
guard (fabricated sentence flagged, grounded sentence passed), obligation/permissive/
info classification, PDF text extraction on a generated minimal PDF fixture, demo-mode
answering with citations, corpus size, and version-pair diffing.

Smoke test (launches the app headlessly and asserts HTTP 200):

```bash
python smoke_test.py
```

## Sample data

The `corpus/` directory contains **SAMPLE DATA**: 12 synthetic documents written in
realistic regulatory language but not copied from any authority (4 GST circular
excerpts including one original/revised version pair, 3 RBI KYC master direction
excerpts, 2 FSSAI packaging/labeling excerpts, 2 SEBI LODR excerpts and 1
data-protection consent notice). Nothing here is legal advice.

## Honest limitations

- The bundled corpus is tiny (12 synthetic documents); retrieval quality in production
  would need the real corpus and stronger retrievers (dense embeddings, rerankers).
- The demo engine quotes rather than synthesizes; answers are faithful but not
  paraphrased, so phrasing can feel repetitive.
- The citation guard is lexical (n-gram overlap): it can miss paraphrased claims and
  cannot judge logical entailment. It is a triage signal, not a proof of correctness.
- PDF parsing relies on pypdf and works best on text-based PDFs; scanned PDFs yield
  little or no text.
- Page numbers are estimated from word positions, not extracted from the source layout.
- Sentence classification is keyword-based; nuanced obligations (e.g., conditional
  permissions) may be misclassified.
- The diff is sentence-level with a 0.5 similarity threshold for pairing Modified
  sentences; heavily restructured documents may produce noisier Added/Removed pairs.
