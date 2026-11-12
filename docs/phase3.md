# Phase 3: Hybrid search and reranking

Built on the Phase 2 semantic index, so there is nothing to re-ingest.

## The retrieval pipeline
1. **Dense search**: top 30 chunks by bge-base embedding similarity (meaning).
2. **Keyword search (BM25)**: top 30 chunks by exact-word overlap, weighted by how rare each word
   is. Catches names, numbers and terms like "Rohan", "92nd", "₹84,000", "RAG-Sequence".
3. **Reciprocal rank fusion**: each list votes 1/(60 + rank) for its chunks; chunks both lists
   like rise to the top. No score calibration needed.
4. **Cross-encoder reranking** (RerankSystem only): ms-marco-MiniLM-L-6-v2 reads the question
   and each candidate together and scores how well it answers. Far more precise than comparing
   two separate embeddings, but too slow to run on the whole corpus, hence steps 1–3.
5. **Diversity cap**: at most 3 chunks per document version in the final 5.

## Systems
| Name | Steps | Run |
|---|---|---|
| `semantic` | 1 | `pka.semantic:SemanticSystem` |
| `hybrid` | 1, 2, 3, 5 | `pka.hybrid:HybridSystem` |
| `hybrid_rerank` | 1–5 | `pka.hybrid:RerankSystem` |

## Commands
```bash
python -m eval.run_eval --system pka.hybrid:HybridSystem --retrieval-only
python -m eval.run_eval --system pka.hybrid:RerankSystem --retrieval-only
python -m eval.compare semantic_retrieval hybrid_retrieval
python -m eval.compare semantic_retrieval hybrid_rerank_retrieval
python -m eval.run_eval --system pka.hybrid:RerankSystem --judge     # once, at the end
python -m pka.ask "What percentile did Rohan suggest?"                # uses the reranker
```

## Tuning (in .env, no re-ingest needed)
`CANDIDATE_POOL` (30), `MAX_PER_DOC` (3), `RERANKER_MODEL`
(try `BAAI/bge-reranker-base`: stronger, but ~1 GB and slower on a laptop).
