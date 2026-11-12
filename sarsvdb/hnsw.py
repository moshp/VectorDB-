"""Hierarchical Navigable Small World index (Malkov & Yashunin, 2018).

Distances are cosine distances, 1 - dot(a, b), on unit-length vectors.
Deleted nodes stay in the graph as stepping stones but never appear in results.
"""
import heapq
import json
import math
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class HNSW:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100, seed: int = 42):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.rng = random.Random(seed)
        self.data = np.empty((0, dim), dtype=np.float32)
        self.count = 0
        self.levels: list[int] = []
        self.layers: list[dict[int, list[int]]] = []
        self.entry: Optional[int] = None
        self.max_level = -1
        self.deleted: set[int] = set()

    def __len__(self) -> int:
        return self.count - len(self.deleted)

    def _dist(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        return 1.0 - self.data[ids] @ q

    def _reserve(self, n: int) -> None:
        if n <= len(self.data):
            return
        grown = np.empty((max(n, 2 * len(self.data), 1024), self.dim), dtype=np.float32)
        grown[: self.count] = self.data[: self.count]
        self.data = grown

    def _search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int,
                      admit: Optional[Callable[[int], bool]] = None) -> list[tuple[float, int]]:
        """Best-first search on one layer. Every visited node can be traversed,
        but only nodes passing admit() enter the result set."""
        graph = self.layers[layer]
        visited = set(entry_points)
        dists = self._dist(q, entry_points)
        candidates = [(float(d), e) for d, e in zip(dists, entry_points)]
        heapq.heapify(candidates)
        results: list[tuple[float, int]] = []  # max-heap via negated distance
        for d, e in candidates:
            if admit is None or admit(e):
                heapq.heappush(results, (-d, e))
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            d_c, c = heapq.heappop(candidates)
            if len(results) >= ef and d_c > -results[0][0]:
                break
            fresh = [n for n in graph[c] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for d_n, n in zip(self._dist(q, fresh), fresh):
                d_n = float(d_n)
                if len(results) < ef or d_n < -results[0][0]:
                    heapq.heappush(candidates, (d_n, n))
                    if admit is None or admit(n):
                        heapq.heappush(results, (-d_n, n))
                        if len(results) > ef:
                            heapq.heappop(results)
        return sorted((-d, n) for d, n in results)

    def _select_neighbors(self, base: np.ndarray, candidates: list[tuple[float, int]],
                          m: int) -> list[int]:
        """Heuristic selection: keep a candidate only if it is closer to the base
        than to any neighbor already kept, which spreads links across directions.
        Pruned candidates fill any remaining slots."""
        selected: list[int] = []
        pruned: list[int] = []
        for d, c in sorted(candidates):
            if len(selected) >= m:
                break
            if not selected or d < float(np.min(1.0 - self.data[selected] @ self.data[c])):
                selected.append(c)
            else:
                pruned.append(c)
        for c in pruned:
            if len(selected) >= m:
                break
            selected.append(c)
        return selected

    def add(self, node: int, vector: np.ndarray) -> None:
        """Nodes must be added with consecutive ids starting at 0."""
        if node != self.count:
            raise ValueError(f"expected node id {self.count}, got {node}")
        self._reserve(node + 1)
        self.data[node] = vector
        self.count += 1

        level = int(-math.log(1.0 - self.rng.random()) * self.mL)
        self.levels.append(level)
        while len(self.layers) <= level:
            self.layers.append({})
        for l in range(level + 1):
            self.layers[l][node] = []

        if self.entry is None:
            self.entry, self.max_level = node, level
            return

        ep = [self.entry]
        for l in range(self.max_level, level, -1):
            ep = [self._search_layer(vector, ep, 1, l)[0][1]]

        for l in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(vector, ep, self.ef_construction, l)
            max_links = self.M0 if l == 0 else self.M
            neighbors = self._select_neighbors(vector, found, self.M)
            self.layers[l][node] = neighbors
            for n in neighbors:
                links = self.layers[l][n]
                links.append(node)
                if len(links) > max_links:
                    base = self.data[n]
                    scored = list(zip((1.0 - self.data[links] @ base).tolist(), links))
                    self.layers[l][n] = self._select_neighbors(base, scored, max_links)
            ep = [n for _, n in found]

        if level > self.max_level:
            self.entry, self.max_level = node, level

    def mark_deleted(self, node: int) -> None:
        self.deleted.add(node)

    def search(self, q: np.ndarray, k: int, ef: int = 64,
               allowed: Optional[Callable[[int], bool]] = None) -> list[tuple[float, int]]:
        """Returns up to k (cosine similarity, node) pairs, best first."""
        if self.entry is None:
            return []
        ep = [self.entry]
        for l in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]

        def admit(n: int) -> bool:
            return n not in self.deleted and (allowed is None or allowed(n))

        found = self._search_layer(q, ep, max(ef, k), 0, admit)
        return [(1.0 - d, n) for d, n in found[:k]]

    def n_edges(self) -> int:
        return sum(len(v) for layer in self.layers for v in layer.values())

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "count": self.count,
            "levels": self.levels,
            "entry": self.entry,
            "max_level": self.max_level,
            "deleted": sorted(self.deleted),
            "layers": [{str(k): v for k, v in layer.items()} for layer in self.layers],
            "rng_state": [self.rng.getstate()[0], list(self.rng.getstate()[1]), self.rng.getstate()[2]],
        }))

    @classmethod
    def load(cls, path: Path, vectors: np.ndarray) -> "HNSW":
        state = json.loads(path.read_text())
        index = cls(state["dim"], state["M"], state["ef_construction"])
        index._reserve(state["count"])
        index.count = state["count"]
        index.data[: index.count] = vectors[: index.count]
        index.levels = state["levels"]
        index.entry = state["entry"]
        index.max_level = state["max_level"]
        index.deleted = set(state["deleted"])
        index.layers = [{int(k): v for k, v in layer.items()} for layer in state["layers"]]
        version, internal, gauss = state["rng_state"]
        index.rng.setstate((version, tuple(internal), gauss))
        return index
