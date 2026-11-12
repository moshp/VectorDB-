"""Vector store: vectors in a NumPy file, text and metadata in SQLite.

Each chunk owns one row of the vector matrix. Deleting a chunk flags its row;
updating a chunk deletes the old row and appends a new one.
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from .hnsw import HNSW

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    row INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    doc TEXT NOT NULL,
    title TEXT,
    category TEXT,
    url TEXT,
    page_start INTEGER,
    page_end INTEGER,
    n_tokens INTEGER,
    text TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc);
"""
FIELDS = ("chunk_id", "doc", "title", "category", "url", "page_start", "page_end", "n_tokens", "text")


class VectorStore:
    def __init__(self, path: Path, dim: int, model_name: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.model_name = model_name
        self.db = sqlite3.connect(self.path / "chunks.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        vec_file = self.path / "vectors.npy"
        self.vectors = np.load(vec_file) if vec_file.exists() else np.empty((0, dim), dtype=np.float32)
        self.live = np.zeros(len(self.vectors), dtype=bool)
        for (row,) in self.db.execute("SELECT row FROM chunks WHERE deleted = 0"):
            self.live[row] = True
        hnsw_file = self.path / "hnsw.json"
        self.hnsw: Optional[HNSW] = HNSW.load(hnsw_file, self.vectors) if hnsw_file.exists() else None

    @classmethod
    def open(cls, path: Path) -> "VectorStore":
        meta = json.loads((Path(path) / "meta.json").read_text())
        return cls(path, meta["dim"], meta["model_name"])

    def __len__(self) -> int:
        return int(self.live.sum())

    def add(self, chunks: list[dict], vectors: np.ndarray) -> list[int]:
        if len(chunks) != len(vectors) or vectors.shape[1] != self.dim:
            raise ValueError("chunks and vectors must align and match the store dimension")
        start = len(self.vectors)
        rows = list(range(start, start + len(chunks)))
        self.db.executemany(
            f"INSERT INTO chunks (row, {', '.join(FIELDS)}) VALUES (?, {', '.join('?' * len(FIELDS))})",
            [(r, *(c.get(f) for f in FIELDS)) for r, c in zip(rows, chunks)],
        )
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.live = np.concatenate([self.live, np.ones(len(chunks), dtype=bool)])
        if self.hnsw is not None:
            for r, v in zip(rows, vectors):
                self.hnsw.add(r, v)
        return rows

    def delete(self, chunk_ids: list[str]) -> int:
        rows = [r for (r,) in self.db.execute(
            f"SELECT row FROM chunks WHERE deleted = 0 AND chunk_id IN ({', '.join('?' * len(chunk_ids))})",
            chunk_ids,
        )]
        self.db.executemany("UPDATE chunks SET deleted = 1 WHERE row = ?", [(r,) for r in rows])
        for r in rows:
            self.live[r] = False
            if self.hnsw is not None:
                self.hnsw.mark_deleted(r)
        return len(rows)

    def upsert(self, chunks: list[dict], vectors: np.ndarray) -> list[int]:
        self.delete([c["chunk_id"] for c in chunks])
        return self.add(chunks, vectors)

    def build_hnsw(self, M: int, ef_construction: int, seed: int = 42, progress: bool = False) -> None:
        self.hnsw = HNSW(self.dim, M, ef_construction, seed)
        for r in range(len(self.vectors)):
            self.hnsw.add(r, self.vectors[r])
            if not self.live[r]:
                self.hnsw.mark_deleted(r)
            if progress and (r + 1) % 1000 == 0:
                print(f"  HNSW: {r + 1}/{len(self.vectors)} nodes", flush=True)

    def _allowed_rows(self, docs: Optional[list[str]]) -> np.ndarray:
        if not docs:
            return self.live
        mask = np.zeros(len(self.vectors), dtype=bool)
        for (r,) in self.db.execute(
            f"SELECT row FROM chunks WHERE deleted = 0 AND doc IN ({', '.join('?' * len(docs))})", docs
        ):
            mask[r] = True
        return mask

    def search(self, query: np.ndarray, k: int = 5, docs: Optional[list[str]] = None,
               method: str = "flat", ef: int = 64) -> list[dict]:
        """method 'flat' compares the query with every chunk (exact);
        'hnsw' walks the graph index (approximate)."""
        mask = self._allowed_rows(docs)
        if method == "flat":
            scores = self.vectors @ query
            scores[~mask] = -np.inf
            n_allowed = int(mask.sum())
            k = min(k, n_allowed)
            if k == 0:
                return []
            top = np.argpartition(-scores, k - 1)[:k]
            ranked = [(float(scores[r]), int(r)) for r in top[np.argsort(-scores[top])]]
        elif method == "hnsw":
            if self.hnsw is None:
                raise RuntimeError("no HNSW index; run build_hnsw first")
            allowed = None if not docs else (lambda n: bool(mask[n]))
            ranked = self.hnsw.search(query, k, ef=ef, allowed=allowed)
        else:
            raise ValueError(f"unknown method {method!r}")
        return self._hits(ranked)

    def _hits(self, ranked: list[tuple[float, int]]) -> list[dict]:
        hits = []
        for score, row in ranked:
            rec = self.db.execute("SELECT * FROM chunks WHERE row = ?", (row,)).fetchone()
            hits.append({**{f: rec[f] for f in FIELDS}, "row": row, "score": score})
        return hits

    def live_chunks(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM chunks WHERE deleted = 0 ORDER BY row")]

    def save(self) -> None:
        self.db.commit()
        np.save(self.path / "vectors.npy", self.vectors)
        if self.hnsw is not None:
            self.hnsw.save(self.path / "hnsw.json")
        (self.path / "meta.json").write_text(json.dumps({"dim": self.dim, "model_name": self.model_name}))
