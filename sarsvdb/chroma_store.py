"""ChromaDB loaded with the same chunks and vectors, for comparison."""
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np

COLLECTION = "sars_chunks"
META_FIELDS = ("doc", "title", "category", "url", "page_start", "page_end", "n_tokens")


class ChromaStore:
    def __init__(self, path: Path):
        self.client = chromadb.PersistentClient(path=str(path))

    @property
    def collection(self):
        return self.client.get_collection(COLLECTION, embedding_function=None)

    def build(self, chunks: list[dict], vectors: np.ndarray, M: int,
              ef_construction: int, ef_search: int) -> None:
        if COLLECTION in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(COLLECTION)
        col = self.client.create_collection(
            COLLECTION,
            embedding_function=None,
            configuration={"hnsw": {
                "space": "cosine",
                "max_neighbors": M,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            }},
        )
        batch = self.client.get_max_batch_size()
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            col.add(
                ids=[c["chunk_id"] for c in part],
                embeddings=vectors[i:i + batch],
                documents=[c["text"] for c in part],
                metadatas=[{f: c[f] for f in META_FIELDS if c.get(f) is not None} for c in part],
            )

    def search(self, query: np.ndarray, k: int = 5, docs: Optional[list[str]] = None) -> list[dict]:
        where = None
        if docs:
            where = {"doc": docs[0]} if len(docs) == 1 else {"doc": {"$in": docs}}
        res = self.collection.query(
            query_embeddings=query[None, :],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {**meta, "chunk_id": cid, "text": text, "score": 1.0 - dist}
            for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                             res["metadatas"][0], res["distances"][0])
        ]
