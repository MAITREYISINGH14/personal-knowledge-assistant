"""Phase 2 system: semantic chunks with contextual headers, embedded with bge-base-en-v1.5.

Retrieval and answering work exactly like the baseline; only the index differs, so any
change in the eval scores comes from chunking and embeddings.
"""
from pka.baseline import BaselineSystem
from pka.config import settings


class SemanticSystem(BaselineSystem):
    name = "semantic"
    collection = settings.semantic_collection
    embed_model = settings.semantic_embed_model
    query_prefix = settings.semantic_query_prefix
    ingest_hint = "python -m pka.ingest --chunker semantic --reset"
