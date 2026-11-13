"""Qdrant vector index wrapper."""
from qdrant_client import QdrantClient, models

from pka.storage.ids import NAMESPACE, point_id  # noqa: F401  (re-exported)

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

    def search(self, vector, limit: int = 5, query_filter=None):
        return self.client.query_points(self.collection, query=vector, limit=limit,
                                        query_filter=query_filter, with_payload=True).points


def valid_on_filter(day: int):
    """Chunks whose validity period covers the given ordinal day."""
    return models.Filter(must=[
        models.FieldCondition(key="valid_from_day", range=models.Range(lte=day)),
        models.FieldCondition(key="valid_to_day", range=models.Range(gt=day)),
    ])
