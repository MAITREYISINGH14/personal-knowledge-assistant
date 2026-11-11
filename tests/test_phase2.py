"""Phase 2: sentence units, semantic chunking, contextual headers, per-index bookkeeping."""
import json
import re
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

from pka.chunking.semantic import context_header, semantic_chunks
from pka.chunking.sentences import split_sentences, to_units
from pka.ingestion.loaders import Block


def bag_of_words(texts):
    """Stand-in embedder: texts sharing words get similar vectors."""
    out = []
    for t in texts:
        v = np.zeros(256)
        for w in re.findall(r"[a-z]{4,}", t.lower()):
            v[zlib.crc32(w.encode()) % 256] += 1
        n = np.linalg.norm(v)
        out.append(v / n if n else np.full(256, 1 / 16))
    return out


def count(texts):
    return [len(t.split()) for t in texts]


GARDEN = ["Tomatoes need full sun and regular watering in the garden.",
          "Garden soil for tomatoes should be rich in compost.",
          "Prune tomato suckers in the garden every week.",
          "Mulch keeps garden soil moist around tomato roots."]
ENGINE = ["The car engine needs fresh oil every ten thousand kilometres.",
          "Engine oil filters should be changed with the oil.",
          "A noisy engine often means worn belts or low oil.",
          "Check engine coolant levels before long drives."]


class TestSentences(unittest.TestCase):
    def test_abbreviations_do_not_split(self):
        s = split_sentences("We met Dr. Rao today. He suggested e.g. Qdrant. It worked.")
        self.assertEqual(s, ["We met Dr. Rao today.", "He suggested e.g. Qdrant.", "It worked."])

    def test_tables_and_lists_stay_whole(self):
        units = to_units([Block("| a | b |\n|---|---|\n| 1 | 2 |\n- first item\n- second item", "S")])
        self.assertEqual([u.kind for u in units], ["table", "list", "list"])
        self.assertIn("| 1 | 2 |", units[0].text)


class TestSemanticChunker(unittest.TestCase):
    def chunk(self, blocks, **kw):
        params = dict(embed_fn=bag_of_words, count_tokens=count, max_tokens=80, min_tokens=20,
                      tiny_tokens=8, single_doc_tokens=30, percentile=80, window=2)
        params.update(kw)
        return semantic_chunks(blocks, **params)

    def test_cuts_where_topic_changes(self):
        text = " ".join(GARDEN + ENGINE)
        chunks, stats = self.chunk([Block(text, "p.1")], is_pdf=True)
        self.assertEqual(len(chunks), 2)
        self.assertIn("tomato", chunks[0].text.lower())
        self.assertNotIn("tomato", chunks[1].text.lower())
        self.assertNotIn("engine", chunks[0].text.lower())
        self.assertEqual(stats.semantic_cuts, 1)

    def test_headings_are_hard_boundaries(self):
        blocks = [Block(" ".join(GARDEN), "Doc > Garden"), Block(" ".join(ENGINE), "Doc > Car")]
        chunks, stats = self.chunk(blocks, percentile=100)
        self.assertEqual([c.section for c in chunks], ["Doc > Garden", "Doc > Car"])
        self.assertEqual(stats.structural_cuts, 1)

    def test_short_document_kept_whole(self):
        chunks, stats = self.chunk([Block("A short note. Just two lines.", "")], single_doc_tokens=100)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(stats.single_chunk_doc)

    def test_continuity_penalty_protects_however(self):
        # The engine sentence after "However" leans on the garden text; with a strong
        # penalty the cut moves off that gap.
        text = " ".join(GARDEN + ["However, the engine of the tiller needs oil too."] + ENGINE[1:])
        chunks, _ = self.chunk([Block(text, "p.1")], is_pdf=True, w_cont=1.0)
        for c in chunks:
            self.assertFalse(c.text.startswith("However"))

    def test_oversized_pieces_are_split(self):
        text = " ".join(GARDEN * 6)   # one topic, far over max_tokens
        chunks, _ = self.chunk([Block(text, "p.1")], is_pdf=True, percentile=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.n_tokens <= 80 for c in chunks))

    def test_tiny_section_merges(self):
        blocks = [Block(" ".join(GARDEN), "Doc > Garden"), Block("Signed.", "Doc > Sign")]
        chunks, _ = self.chunk(blocks)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Doc > Sign", chunks[0].section)


class TestHeader(unittest.TestCase):
    def test_header(self):
        doc = mock.Mock(title="Rental Agreement", doc_date="2026-11-01", versioned=True, version=2)
        h = context_header(doc, "Rental Agreement > 4. Licence Fee | Rental Agreement > 5. Deposit")
        self.assertEqual(h, "Document: Rental Agreement | Section: 4. Licence Fee; 5. Deposit "
                            "| Dated: 2026-11-01 | Version 2")
        pdf = mock.Mock(title="RAPTOR", doc_date=None, versioned=False, version=1)
        self.assertEqual(context_header(pdf, "p.3 | p.4"), "Document: RAPTOR")


class TestMetadataPerIndex(unittest.TestCase):
    def test_indexes_tracked_separately(self):
        from pka.storage.metadata import MetadataStore
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "m.db")
            doc = mock.Mock(doc_id="d", version=1, path="p", sha256="abc", title="t",
                            doc_date=None, file_mtime="x")
            store.save_document(doc, "pka_baseline", [])
            self.assertEqual(store.stored_hash("d", 1, "pka_baseline"), "abc")
            self.assertIsNone(store.stored_hash("d", 1, "pka_semantic"))   # not yet built


class TestCompare(unittest.TestCase):
    def test_compare_runs(self):
        from eval import compare
        def report(system, hit):
            s = {"hit_rate": hit, "mrr": hit, "correctness": hit, "full_hit_rate": hit, "n": 1}
            return {"system": system, "llm": "x", "summary": {"fact": s, "all": s},
                    "rows": [{"id": "q1", "hit": hit > 0.5}]}
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_text(json.dumps(report("baseline", 0.4)))
            b.write_text(json.dumps(report("semantic", 0.9)))
            out = []
            with mock.patch("sys.argv", ["compare", str(a), str(b)]), \
                 mock.patch("builtins.print", side_effect=lambda *x: out.append(" ".join(map(str, x)))):
                compare.main()
        text = "\n".join(out)
        self.assertIn("▲", text)
        self.assertIn("q1: MISS → HIT", text)


if __name__ == "__main__":
    unittest.main()
