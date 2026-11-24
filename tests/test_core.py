"""RegIntel core test suite (unittest).

Run from the project root:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import engine  # noqa: E402
import versioning  # noqa: E402

CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")


class ChunkerTests(unittest.TestCase):
    def test_chunker_metadata_correctness(self):
        path = os.path.join(CORPUS_DIR, "gst_circular_178_2022.txt")
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        chunks = engine.chunk_doc(text, doc_id="gst_circular_178_2022",
                                  title="GST Circular No. 178/2022")
        self.assertTrue(chunks)
        for c in chunks:
            self.assertEqual(c.doc_id, "gst_circular_178_2022")
            self.assertEqual(c.title, "GST Circular No. 178/2022")
            self.assertTrue(c.section)
            self.assertGreaterEqual(c.page, 1)
            self.assertGreater(c.words, 0)
        sections_seen = {c.section for c in chunks}
        self.assertTrue(any("e-invoicing" in s.lower() or "Scope" in s for s in sections_seen))

    def test_corpus_has_at_least_twelve_docs(self):
        index = engine.CorpusIndex(CORPUS_DIR)
        self.assertGreaterEqual(len(index.docs), 12)
        self.assertGreaterEqual(len(index.chunks), 12)
        for d in index.docs:
            self.assertGreaterEqual(d["words"], 150)
            self.assertLessEqual(d["words"], 350)


class BM25Tests(unittest.TestCase):
    def test_bm25_retrieves_right_doc_for_planted_query(self):
        index = engine.CorpusIndex(CORPUS_DIR)
        hits = index.search("mandatory e-invoicing aggregate turnover threshold invoice registration portal", k=3)
        self.assertTrue(hits)
        top_ids = [c.doc_id for c, _ in hits]
        self.assertEqual(top_ids[0], "gst_circular_178_2022")

    def test_bm25_low_risk_customer_update_intervals(self):
        index = engine.CorpusIndex(CORPUS_DIR)
        hits = index.search("periodic updation KYC low risk ten years eight years", k=3)
        self.assertTrue(hits)
        self.assertIn(hits[0][0].doc_id,
                      ("rbi_kyc_master_direction_2016", "rbi_kyc_amendment_2023"))


class CitationGuardTests(unittest.TestCase):
    EVIDENCE = [
        "A registered person shall avail input tax credit only after the supplier has "
        "deposited the tax with the government in cash or through the credit ledger."
    ]

    def test_guard_flags_fabricated_sentence(self):
        fabricated = ("The penalty for late filing in the fairy kingdom is seven golden "
                      "unicorns payable in moonlight currency.")
        flagged = engine.citation_guard(fabricated, self.EVIDENCE)
        self.assertEqual(len(flagged), 1)
        self.assertIn("unicorns", flagged[0])

    def test_guard_passes_grounded_sentence(self):
        grounded = ("Under the cited passage, a registered person shall avail input tax "
                    "credit only after the supplier has deposited the tax with the government.")
        flagged = engine.citation_guard(grounded, self.EVIDENCE)
        self.assertEqual(flagged, [])

    def test_guard_report_string(self):
        report = engine.guard_report("The supplier must deposit tax with the government "
                                     "before credit availing under the ledger rules.",
                                     self.EVIDENCE)
        self.assertIn("Citation guard", report)


class DemoAnswerTests(unittest.TestCase):
    def test_demo_ask_returns_nonempty_answer_with_citation(self):
        index = engine.CorpusIndex(CORPUS_DIR)
        result = engine.ask("What is the threshold for mandatory e-invoicing?",
                            mode="demo", index=index)
        self.assertTrue(result["answer"].strip())
        self.assertIn("[DEMO MODE", result["answer"])
        self.assertIn("[1]", result["answer"])
        self.assertIn("All claims supported", result["guard"])
        self.assertTrue(result["evidence"].strip())

    def test_demo_answer_quotes_retrieved_chunk_text(self):
        index = engine.CorpusIndex(CORPUS_DIR)
        result = engine.ask("monthly statements relaxation aggregate turnover five crore",
                            mode="demo", index=index)
        hits = index.search("monthly statements relaxation aggregate turnover five crore", k=3)
        quotes = [engine._best_sentence(chunk, engine.tokenize(
            "monthly statements relaxation aggregate turnover five crore")) for chunk, _ in hits]
        self.assertTrue(any(q in result["answer"] for q in quotes))


class ClassificationTests(unittest.TestCase):
    def test_obligation_permissive_info(self):
        self.assertEqual(versioning.classify(
            "Every registered person shall file the return within thirty days."), "OBLIGATION")
        self.assertEqual(versioning.classify(
            "The registered person may opt for the quarterly scheme if eligible."), "PERMISSIVE")
        self.assertEqual(versioning.classify(
            "This circular supersedes the earlier circular issued in July."), "INFO")

    def test_obligation_keywords(self):
        for word in ("must", "required", "penalty", "mandatory", "liable"):
            self.assertEqual(versioning.classify(f"The entity {word} to comply."), "OBLIGATION")


class DiffTests(unittest.TestCase):
    def test_compare_bundled_version_pair_detects_changes(self):
        with open(os.path.join(CORPUS_DIR, "gst_circular_12_2023.txt"), encoding="utf-8") as fh:
            a = fh.read()
        with open(os.path.join(CORPUS_DIR, "gst_circular_12_2023r.txt"), encoding="utf-8") as fh:
            b = fh.read()
        rows, counts = versioning.compare_versions(a, b)
        self.assertGreaterEqual(counts["TOTAL"], 3)
        self.assertGreater(counts["OBLIGATION"], 0)
        types = {r["type"] for r in rows}
        self.assertTrue(types & {"Added", "Modified", "Removed"})

    def test_compare_reports_added_removed_modified(self):
        a = "The officer shall issue a notice within ten days. The process may be digital."
        b = "The officer shall issue a notice within seven days. The officer must record reasons."
        rows, counts = versioning.compare_versions(a, b)
        types = {r["type"] for r in rows}
        self.assertIn("Modified", types)
        self.assertIn("Added", types)
        self.assertIn("Removed", types)
        self.assertEqual(counts["TOTAL"], 3)
        modified = [r for r in rows if r["type"] == "Modified"]
        self.assertEqual(modified[0]["classification"], "OBLIGATION")
        added = [r for r in rows if r["type"] == "Added"]
        self.assertEqual(added[0]["classification"], "OBLIGATION")
        removed = [r for r in rows if r["type"] == "Removed"]
        self.assertEqual(removed[0]["classification"], "PERMISSIVE")


class PdfExtractionTests(unittest.TestCase):
    def test_pdf_text_extraction_on_generated_pdf(self):
        lines = ["RegIntel test document",
                 "Every registered person shall file the return within thirty days.",
                 "The registered person may opt for the quarterly scheme."]
        pdf_bytes = versioning.build_simple_pdf(lines)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        text = versioning.extract_pdf_text(pdf_bytes)
        for line in lines:
            self.assertIn(line.replace("(", "").replace(")", ""), text.replace("\n", " "))

    def test_classification_after_pdf_extraction(self):
        pdf_bytes = versioning.build_simple_pdf(
            ["1. General", "A food business operator shall use only compliant packaging materials."])
        text = versioning.extract_pdf_text(pdf_bytes)
        self.assertEqual(versioning.classify(text), "OBLIGATION")


if __name__ == "__main__":
    unittest.main()
