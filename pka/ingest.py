"""Ingest the corpus into an index.

  python -m pka.ingest --reset                       # Phase 1 baseline (fixed-size chunks)
  python -m pka.ingest --chunker semantic --reset    # Phase 2 semantic chunks
Without --reset, only new or changed files are processed.
"""
import argparse
import time

from pka.config import settings
from pka.embeddings import Embedder
from pka.ingestion.loaders import discover, load_file
from pka.storage.metadata import MetadataStore
from pka.storage.vectors import VectorStore, point_id


def fixed_plan():
    from pka.chunking.fixed import fixed_size_chunks
    embedder = Embedder(settings.embed_model)

    def chunk(doc):
        chunks = fixed_size_chunks(doc.blocks, settings.baseline_chunk_words,
                                   settings.baseline_overlap_words)
        return [(c, c.text) for c in chunks], None   # baseline embeds the bare text

    return settings.baseline_collection, embedder, chunk


def semantic_plan():
    from pka.chunking.semantic import context_header, semantic_chunks
    embedder = Embedder(settings.semantic_embed_model)
    boundary = Embedder(settings.boundary_model)

    def chunk(doc):
        chunks, stats = semantic_chunks(
            doc.blocks, embed_fn=boundary.embed, count_tokens=embedder.count_tokens_batch,
            is_pdf=doc.path.suffix.lower() == ".pdf", max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens, percentile=settings.chunk_percentile,
            window=settings.chunk_window)
        return [(c, context_header(doc, c.section) + "\n\n" + c.text) for c in chunks], stats

    return settings.semantic_collection, embedder, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=["fixed", "semantic"], default="fixed")
    ap.add_argument("--reset", action="store_true", help="drop this index and rebuild it")
    args = ap.parse_args()
    t0 = time.perf_counter()

    print("Loading models (the first run downloads them)...")
    coll, embedder, chunk = fixed_plan() if args.chunker == "fixed" else semantic_plan()
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
    added = skipped = failed = total_chunks = over_limit = 0
    cuts = {"semantic": 0, "structural": 0, "forced": 0, "single": 0}
    for path in files:
        try:
            doc = load_file(path)
        except Exception as e:      # a broken file shouldn't stop the run
            print(f"  FAIL  {path.name}: {e}")
            failed += 1
            continue
        if meta.stored_hash(doc.doc_id, doc.version, coll) == doc.sha256:
            skipped += 1
            continue
        pairs, stats = chunk(doc)
        if not pairs:
            print(f"  WARN  {path.name}: no text extracted (scanned PDF?)")
            failed += 1
            continue
        vectors = embedder.embed([t for _, t in pairs])
        token_counts = embedder.count_tokens_batch([t for _, t in pairs])
        ids, payloads, rows = [], [], []
        for (c, embed_text), n_tok in zip(pairs, token_counts):
            over_limit += n_tok > embedder.max_tokens
            pid = point_id(coll, doc.doc_id, doc.version, c.index)
            ids.append(pid)
            payloads.append({"doc_id": doc.doc_id, "version": doc.version, "section": c.section,
                             "chunk_index": c.index, "text": c.text, "embed_text": embed_text,
                             "doc_date": doc.doc_date, "title": doc.title})
            rows.append({"chunk_id": pid, "chunk_index": c.index, "section": c.section,
                         "text": c.text, "n_words": c.n_words, "n_tokens": n_tok})
        store.delete_document(doc.doc_id, doc.version)
        store.upsert(ids, vectors, payloads)
        meta.save_document(doc, coll, rows)
        total_chunks += len(pairs)
        added += 1
        extra = ""
        if stats:
            cuts["semantic"] += stats.semantic_cuts
            cuts["structural"] += stats.structural_cuts
            cuts["forced"] += stats.forced_cuts
            cuts["single"] += stats.single_chunk_doc
            extra = (" (kept whole)" if stats.single_chunk_doc else
                     f" (cuts: {stats.structural_cuts} heading, {stats.semantic_cuts} semantic, "
                     f"{stats.forced_cuts} size)")
        print(f"  ok    {doc.doc_id} v{doc.version}: {len(pairs)} chunks{extra}")

    docs, chunks_in_db = meta.counts(coll)
    print(f"\nIndex '{coll}': ingested {added} files ({total_chunks} chunks), "
          f"unchanged {skipped}, failed {failed}.")
    print(f"It now holds {docs} document versions and {chunks_in_db} chunks.")
    if total_chunks:
        print(f"Chunks over the model's {embedder.max_tokens}-token limit: {over_limit} "
              f"({100 * over_limit / total_chunks:.1f}%)")
    if args.chunker == "semantic" and added:
        total_cuts = cuts["semantic"] + cuts["structural"] + cuts["forced"]
        share = 100 * cuts["semantic"] / total_cuts if total_cuts else 0
        print(f"Cuts: {cuts['structural']} at headings, {cuts['semantic']} semantic, "
              f"{cuts['forced']} for size; {cuts['single']} short docs kept whole. "
              f"Semantic share of cuts: {share:.0f}%")
    print(f"Done in {time.perf_counter() - t0:.0f}s.")


if __name__ == "__main__":
    main()
