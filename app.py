"""RegIntel - Citation-Grounded Regulatory Change Intelligence.

Gradio Blocks application with three tabs:
  1. Ask the Corpus   - BM25 retrieval, demo/live answering, citation guard, evidence panel
  2. Compare Versions - sentence-level diff and obligation classification
  3. Corpus Browser   - corpus stats and document preview
"""

from __future__ import annotations

import os

import gradio as gr

import engine
import versioning

HAS_LLM_KEY = bool(os.environ.get("OPENAI_API_KEY"))
MODE_CHOICES = ["Demo (offline, deterministic)", "Live (LLM)"]
DEFAULT_MODE = MODE_CHOICES[1] if HAS_LLM_KEY else MODE_CHOICES[0]

EXAMPLE_QUESTIONS = [
    "What is the threshold for mandatory e-invoicing?",
    "When may a registered person avail input tax credit?",
    "How often must KYC records be updated for high risk customers?",
    "What are the rules for allergen declaration on food labels?",
]


# ---------------------------------------------------------------------------
# Tab 1: Ask the Corpus
# ---------------------------------------------------------------------------

def run_ask(question: str, mode_label: str) -> tuple:
    question = (question or "").strip()
    if not question:
        return ("Enter a question to query the regulatory corpus.",
                "Citation guard: nothing to verify.",
                "")
    mode = "live" if mode_label.startswith("Live") else "demo"
    result = engine.ask(question, mode=mode)
    answer = result["answer"]
    if result["notice"]:
        answer = f"> **Notice:** {result['notice']}\n\n{answer}"
    if result["mode"] == "live":
        answer = f"[LIVE MODE — LLM synthesis, guarded against unsupported claims]\n\n{answer}"
    return answer, result["guard"], result["evidence"]


# ---------------------------------------------------------------------------
# Tab 2: Compare Versions
# ---------------------------------------------------------------------------

def _file_text(file_value) -> str:
    """Extract text from a gradio File value (str path or FileData/dict)."""
    if not file_value:
        return ""
    path = None
    if isinstance(file_value, str):
        path = file_value
    elif isinstance(file_value, dict):
        path = file_value.get("path") or file_value.get("name")
    else:
        path = getattr(file_value, "path", None) or getattr(file_value, "name", None)
    if not path:
        return ""
    try:
        return versioning.read_upload(path)
    except Exception as exc:
        return f"[Could not read file: {type(exc).__name__}: {exc}]"


def run_compare(file_a, text_a, file_b, text_b) -> tuple:
    content_a = _file_text(file_a) or (text_a or "").strip()
    content_b = _file_text(file_b) or (text_b or "").strip()
    if not content_a or not content_b:
        return ([], "Provide both versions: upload a .txt/.pdf file or paste the text in each box.",
                "")
    rows, counts = versioning.compare_versions(content_a, content_b)
    table = [[r["type"], r["classification"], r["old"], r["new"]] for r in rows]
    summary = versioning.summary_counts_md(counts)
    if not rows:
        summary += "\n\nNo sentence-level changes detected between the two versions."
    return table, summary, "ok"


# ---------------------------------------------------------------------------
# Tab 3: Corpus Browser
# ---------------------------------------------------------------------------

def doc_rows() -> list:
    return engine.get_index().doc_table()


def preview_by_row(evt: gr.SelectData) -> str:
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    docs = engine.get_index().doc_table()
    if 0 <= idx < len(docs):
        return engine.get_index().get_doc_text(docs[idx][0])
    return "Select a document to preview its text."


def preview_by_id(doc_id: str) -> str:
    if not doc_id:
        return "Select a document to preview its text."
    text = engine.get_index().get_doc_text(doc_id)
    return text or "Document not found."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_blocks() -> gr.Blocks:
    with gr.Blocks(title="RegIntel — Citation-Grounded Regulatory Change Intelligence") as demo:
        gr.Markdown(
            "# RegIntel — Citation-Grounded Regulatory Change Intelligence\n"
            "Every answer sentence must carry an inline citation to retrieved evidence. "
            "The citation guard flags unsupported claims; the version-compare tool "
            "classifies Added / Modified / Removed obligations between two circular versions."
        )

        with gr.Tab("Ask the Corpus"):
            mode_radio = gr.Radio(MODE_CHOICES, value=DEFAULT_MODE,
                                  label="Answer mode",
                                  info="Demo mode assembles a deterministic summary strictly from "
                                       "retrieved evidence. Live mode calls an LLM and then runs the "
                                       "citation guard (falls back to demo on failure).")
            question_box = gr.Textbox(label="Question", placeholder="e.g. What is the threshold for mandatory e-invoicing?",
                                      lines=2)
            ask_btn = gr.Button("Ask the Corpus", variant="primary")
            answer_md = gr.Markdown("The answer will appear here.")
            guard_md = gr.Markdown("Citation guard: nothing to verify.")
            gr.Markdown("### Evidence")
            evidence_md = gr.Markdown()
            gr.Examples(EXAMPLE_QUESTIONS, inputs=[question_box], label="Example questions")

            ask_btn.click(run_ask, inputs=[question_box, mode_radio],
                          outputs=[answer_md, guard_md, evidence_md])
            question_box.submit(run_ask, inputs=[question_box, mode_radio],
                                outputs=[answer_md, guard_md, evidence_md])

        with gr.Tab("Compare Versions"):
            gr.Markdown(
                "Compare two versions of a circular. Changed sentences are classified as "
                "**OBLIGATION** (shall / must / required / penalty / mandatory), **PERMISSIVE** "
                "(may / optional / permitted) or **INFO**."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Version A (older)**")
                    file_a = gr.File(file_types=[".txt", ".pdf"], label="Upload Version A (.txt or .pdf)")
                    text_a = gr.Textbox(label="... or paste Version A text", lines=8)
                with gr.Column():
                    gr.Markdown("**Version B (newer)**")
                    file_b = gr.File(file_types=[".txt", ".pdf"], label="Upload Version B (.txt or .pdf)")
                    text_b = gr.Textbox(label="... or paste Version B text", lines=8)
            compare_btn = gr.Button("Compare Versions", variant="primary")
            counts_md = gr.Markdown("Run a comparison to see change counts.")
            diff_table = gr.Dataframe(
                headers=["Change", "Class", "Version A sentence", "Version B sentence"],
                label="Sentence-level diff",
                wrap=True,
            )
            compare_btn.click(run_compare, inputs=[file_a, text_a, file_b, text_b],
                              outputs=[diff_table, counts_md])

        with gr.Tab("Corpus Browser"):
            gr.Markdown(
                "Bundled sample corpus (SAMPLE DATA — synthetic regulatory excerpts for the demo). "
                "Click a row or pick a document below to preview its full text."
            )
            docs_table = gr.Dataframe(
                value=doc_rows(),
                headers=["Doc ID", "Title", "Words", "Sections", "Chunks"],
                label="Corpus documents",
                interactive=False,
            )
            doc_dropdown = gr.Dropdown(choices=[d[0] for d in doc_rows()],
                                       label="Select a document by ID")
            preview_box = gr.Textbox(label="Document preview", lines=20, interactive=False)
            docs_table.select(preview_by_row, inputs=None, outputs=[preview_box])
            doc_dropdown.change(preview_by_id, inputs=[doc_dropdown], outputs=[preview_box])

        gr.Markdown(
            "---\n"
            "*RegIntel demo. The bundled corpus contains synthetic sample documents "
            "(SAMPLE DATA) modelled on public regulatory language; it is not legal advice.*"
        )

    return demo


demo = build_blocks()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
