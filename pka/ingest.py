"""Ingest the corpus into the baseline index.

  python -m pka.ingest            # add new or changed files
  python -m pka.ingest --reset    # rebuild everything from scratch
"""
import argparse
import time

from pka.chunking.fixed import fixed_size_chunks
from pka.config import settings
from pka.embeddings import Embedder
from pka.ingestion.loaders import discover, load_file
from pka.storage.metadata import MetadataStore
from pka.storage.vectors import VectorStore, point_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop the index and re-ingest everything")
    args = ap.parse_args()
    t0 = time.perf_counter()
    coll = settings.baseline_collection

    print(f"Loading embedding model {settings.embed_model} (first run downloads it)...")
    embedder = Embedder(settings.embed_model)
    store = VectorStore(settings.qdrant_url, coll, embedder.dim)
    try:
        store.ensure(reset=args.reset)
    except Exception as e:
        raise SystemExit(f"Can't reach Qdrant at {settings.qdrant_url} ({type(e).__name__}). "
                         "Open Docker Desktop, wait for 'Engine running', then run: docker compose up -d")
    meta = MetadataStore(settings.sqlite_path)
    if args.reset:
        meta.reset(coll)

    files = discover(settings.corpus_dir)
    if not files:
        print(f"No documents found in {settings.corpus_dir}")
        return
    added = skipped = failed = 0
    over_limit = total_chunks = 0
    for path in files:
        try:
            doc = load_file(path)
        except Exception as e:  # a broken file shouldn't stop the run
            print(f"  FAIL  {path.name}: {e}")
            failed += 1
            continue
        if meta.stored_hash(doc.doc_id, doc.version) == doc.sha256:
            skipped += 1
            continue
        chunks = fixed_size_chunks(doc.blocks, settings.baseline_chunk_words,
                                   settings.baseline_overlap_words)
        if not chunks:
            print(f"  WARN  {path.name}: no text extracted (scanned PDF?)")
            failed += 1
            continue
        vectors = embedder.embed([c.text for c in chunks])
        ids, payloads, rows = [], [], []
        for c in chunks:
            n_tok = embedder.count_tokens(c.text)
            over_limit += n_tok > embedder.max_tokens
            pid = point_id(coll, doc.doc_id, doc.version, c.index)
            ids.append(pid)
            payloads.append({"doc_id": doc.doc_id, "version": doc.version, "section": c.section,
                             "chunk_index": c.index, "text": c.text, "doc_date": doc.doc_date,
                             "title": doc.title})
            rows.append({"chunk_id": pid, "chunk_index": c.index, "section": c.section,
                         "text": c.text, "n_words": c.n_words, "n_tokens": n_tok})
        store.delete_document(doc.doc_id, doc.version)
        store.upsert(ids, vectors, payloads)
        meta.save_document(doc, coll, rows)
        total_chunks += len(chunks)
        added += 1
        date = f", dated {doc.doc_date}" if doc.doc_date else ""
        print(f"  ok    {doc.doc_id} v{doc.version}: {len(chunks)} chunks{date}")

    docs, chunks_in_db = meta.counts(coll)
    print(f"\nIngested {added} files ({total_chunks} chunks), unchanged {skipped}, failed {failed}.")
    print(f"Index now holds {docs} document versions and {chunks_in_db} chunks.")
    if total_chunks:
        pct = 100 * over_limit / total_chunks
        print(f"Chunks longer than the model's {embedder.max_tokens}-token limit: "
              f"{over_limit} ({pct:.1f}%) - their ends are truncated when embedded.")
    print(f"Done in {time.perf_counter() - t0:.0f}s.")


if __name__ == "__main__":
    main()
