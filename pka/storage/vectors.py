"""Qdrant vector index wrapper."""
import uuid

from qdrant_client import QdrantClient, models

NAMESPACE = uuid.UUID("6f1c2a52-6b5e-4a51-9a7b-6a1f3e0c9d11")


def point_id(collection: str, doc_id: str, version: int, index: int) -> str:
    """Deterministic id, so re-ingesting the same chunk overwrites instead of duplicating."""
    return str(uuid.uuid5(NAMESPACE, f"{collection}/{doc_id}/v{version}/{index}"))


class VectorStore:
    def __init__(self, url: str, collection: str, dim: int):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim

    def ensure(self, reset: bool = False):
        exists = self.client.collection_exists(self.collection)
        if exists and reset:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE))
            for field, schema in (("doc_id", models.PayloadSchemaType.KEYWORD),
                                  ("version", models.PayloadSchemaType.INTEGER)):
                self.client.create_payload_index(self.collection, field, schema)

    def delete_document(self, doc_id: str, version: int):
        self.client.delete(self.collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="version", match=models.MatchValue(value=version)),
            ])))

    def upsert(self, ids, vectors, payloads, batch: int = 128):
        for i in range(0, len(ids), batch):
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=pid, vector=vec, payload=pl)
                for pid, vec, pl in zip(ids[i:i + batch], vectors[i:i + batch], payloads[i:i + batch])])

    def search(self, vector, limit: int = 5):
        return self.client.query_points(self.collection, query=vector, limit=limit,
                                        with_payload=True).points
