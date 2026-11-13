"""Deterministic point ids (kept free of heavy imports so they are easy to test)."""
import uuid

NAMESPACE = uuid.UUID("6f1c2a52-6b5e-4a51-9a7b-6a1f3e0c9d11")


def point_id(collection: str, doc_id: str, version: int, index: int) -> str:
    """Deterministic id, so re-ingesting the same chunk overwrites instead of duplicating."""
    return str(uuid.uuid5(NAMESPACE, f"{collection}/{doc_id}/v{version}/{index}"))
