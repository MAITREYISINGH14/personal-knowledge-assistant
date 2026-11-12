"""Sentence-transformer embeddings, loaded lazily (the first load downloads the model)."""
import logging
import os
import warnings
from functools import cached_property


def _quiet_libraries():
    """Hide harmless start-up noise from PyTorch and Hugging Face."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("torch.distributed.elastic", "torch.distributed", "huggingface_hub",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self):
        _quiet_libraries()
        from sentence_transformers import SentenceTransformer
        try:   # already downloaded? load it without touching the network
            return SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            return SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))

    def count_tokens_batch(self, texts) -> list:
        ids = self.model.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(x) for x in ids]

    def embed(self, texts, show_progress: bool = False, prefix: str = ""):
        texts = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=show_progress)
        return [v.tolist() for v in vecs]
