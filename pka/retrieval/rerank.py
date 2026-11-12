"""Cross-encoder reranker: reads the question and a passage TOGETHER and scores how well
the passage answers it. Slower than embeddings, so it only re-scores a short candidate list."""
from functools import cached_property


class Reranker:
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name, self.max_length = model_name, max_length

    @cached_property
    def model(self):
        from pka.embeddings import _quiet_libraries
        _quiet_libraries()
        from sentence_transformers import CrossEncoder
        try:
            return CrossEncoder(self.model_name, max_length=self.max_length, local_files_only=True)
        except Exception:
            return CrossEncoder(self.model_name, max_length=self.max_length)

    def score(self, query: str, passages):
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages],
                                                     batch_size=16, show_progress_bar=False)]
