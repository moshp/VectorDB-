import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()
        self.query_prefix = query_prefix

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def encode(self, texts: list[str], batch_size: int = 32, progress: bool = False) -> np.ndarray:
        """Unit-length float32 vectors, so cosine similarity is a dot product."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        ).astype(np.float32)

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        """Some retrieval models expect an instruction in front of each question."""
        return self.encode([self.query_prefix + q for q in questions])
