"""Phase 1 tests for the parts that need no model, database or network."""
import unittest
from pathlib import Path

from pka.chunking.fixed import fixed_size_chunks
from pka.ingestion.dates import parse_doc_date
from pka.ingestion.loaders import Block, load_file, parse_markdown, parse_name

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


class TestDates(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_doc_date("# T\n**Last updated:** 15 December 2025"), "2025-12-15")
        self.assertEqual(parse_doc_date("**Effective date:** 1 October 2026 (replaces x)"), "2026-10-01")
        self.assertEqual(parse_doc_date("Date: 2026-11-03"), "2026-11-03")
        self.assertIsNone(parse_doc_date("no date here"))


class TestMarkdown(unittest.TestCase):
    def test_heading_paths(self):
        blocks, title = parse_markdown("# Doc\n\nIntro.\n\n## A\n\nText a.\n\n### A1\n\nDeep.\n\n## B\n\nText b.")
        self.assertEqual(title, "Doc")
        self.assertEqual([b.section for b in blocks], ["Doc", "Doc > A", "Doc > A > A1", "Doc > B"])

    def test_code_fence_not_a_heading(self):
        blocks, _ = parse_markdown("# Doc\n\n```\n# not a heading\n```")
        self.assertEqual(blocks[0].section, "Doc")
        self.assertIn("# not a heading", blocks[0].text)

    def test_parse_name(self):
        self.assertEqual(parse_name("rental_agreement__v2"), ("rental_agreement", 2))
        self.assertEqual(parse_name("notes"), ("notes", 1))


class TestFixedChunker(unittest.TestCase):
    def test_sizes_and_overlap(self):
        blocks = [Block(" ".join(f"w{i}" for i in range(100)), "S")]
        chunks = fixed_size_chunks(blocks, max_words=40, overlap=10)
        self.assertEqual([c.n_words for c in chunks], [40, 40, 40])
        self.assertEqual(chunks[1].text.split()[0], "w30")      # 10-word overlap
        self.assertEqual(chunks[-1].text.split()[-1], "w99")

    def test_sections_spanned(self):
        blocks = [Block("a " * 30, "Doc > A"), Block("b " * 30, "Doc > B")]
        chunks = fixed_size_chunks(blocks, max_words=50, overlap=5)
        self.assertEqual(chunks[0].section, "Doc > A | Doc > B")

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            fixed_size_chunks([Block("x")], max_words=10, overlap=10)


@unittest.skipUnless((CORPUS / "versioned" / "rental_agreement__v2.md").exists(), "sample corpus not installed")
class TestSampleCorpus(unittest.TestCase):
    def test_rental_v2(self):
        doc = load_file(CORPUS / "versioned" / "rental_agreement__v2.md")
        self.assertEqual((doc.doc_id, doc.version, doc.doc_date), ("rental_agreement", 2, "2026-11-01"))
        self.assertTrue(any("Pets" in b.section for b in doc.blocks))


if __name__ == "__main__":
    unittest.main()
