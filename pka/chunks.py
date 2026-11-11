"""See how a document was chunked - useful for judging boundary quality by eye.

  python -m pka.chunks rental_agreement --version 2
  python -m pka.chunks call_transcript_design_review_2026-11-03 --chunker fixed
"""
import argparse
import sqlite3

from pka.config import settings

COLLECTIONS = {"fixed": settings.baseline_collection, "semantic": settings.semantic_collection}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--chunker", choices=list(COLLECTIONS), default="semantic")
    ap.add_argument("--full", action="store_true", help="print whole chunks, not just the edges")
    args = ap.parse_args()

    conn = sqlite3.connect(settings.sqlite_path)
    query = ("SELECT version, chunk_index, section, n_tokens, text FROM chunks "
             "WHERE collection=? AND doc_id=?")
    params = [COLLECTIONS[args.chunker], args.doc_id]
    if args.version is not None:
        query += " AND version=?"
        params.append(args.version)
    rows = conn.execute(query + " ORDER BY version, chunk_index", params).fetchall()
    if not rows:
        known = [r[0] for r in conn.execute(
            "SELECT DISTINCT doc_id FROM chunks WHERE collection=? ORDER BY doc_id",
            (COLLECTIONS[args.chunker],))]
        print(f"No chunks for '{args.doc_id}' in the {args.chunker} index. Known documents:")
        print("  " + "\n  ".join(known) if known else "  (none - run the ingest first)")
        return
    for version, idx, section, n_tok, text in rows:
        sec = section.split(" | ")
        sec = " > ".join(sec[0].split(" > ")[-1:]) + (f" (+{len(sec) - 1} more)" if len(sec) > 1 else "")
        print(f"\n━━ v{version} chunk {idx}  ·  {n_tok} tokens  ·  {sec}")
        if args.full or len(text) <= 320:
            print(text)
        else:
            print(f"{text[:150]} … {text[-150:]}")


if __name__ == "__main__":
    main()
