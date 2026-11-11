"""Inventory the test corpus and check it against the Phase 0 targets."""
import re
from collections import Counter, defaultdict

from pka.config import settings

SUPPORTED = {".pdf", ".md", ".txt"}
VERSION_RE = re.compile(r"^(?P<doc_id>.+?)__v(?P<version>\d+)$")


def parse_name(stem: str):
    m = VERSION_RE.match(stem)
    return (m["doc_id"], int(m["version"])) if m else (stem, 1)


def main():
    root = settings.corpus_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "README.md"]
    versions = defaultdict(list)
    skipped, by_ext, sizes = [], Counter(), []
    for p in files:
        if p.suffix.lower() not in SUPPORTED:
            skipped.append(p)
            continue
        doc_id, v = parse_name(p.stem)
        versions[doc_id].append(v)
        by_ext[p.suffix.lower()] += 1
        sizes.append(p.stat().st_size)

    multi = {d: sorted(v) for d, v in versions.items() if len(v) > 1}
    print(f"Corpus: {root}")
    print(f"Documents: {len(versions)}  (files: {sum(by_ext.values())})")
    print("By type:  " + (", ".join(f"{e}={n}" for e, n in sorted(by_ext.items())) or "none"))
    if sizes:
        print(f"Size:     smallest {min(sizes)/1024:.0f} KB, largest {max(sizes)/1024:.0f} KB")
    print(f"Versioned docs: {len(multi)}")
    for d, v in sorted(multi.items()):
        gaps = sorted(set(range(1, max(v) + 1)) - set(v))
        print(f"  - {d}: versions {v}" + (f"  (missing {gaps})" if gaps else ""))
    if skipped:
        print(f"Skipped (unsupported): {len(skipped)} e.g. {skipped[0].name}")

    print("\nPhase 0 targets:")
    n = len(versions)
    print(f"  [{'x' if 20 <= n <= 40 else ' '}] 20–40 documents (have {n})")
    print(f"  [{'x' if len(multi) >= 3 else ' '}] 3+ documents with two versions (have {len(multi)})")
    print(f"  [{'x' if '.pdf' in by_ext else ' '}] at least one PDF")


if __name__ == "__main__":
    main()
