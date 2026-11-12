"""Compares exact search, our HNSW index and ChromaDB on retrieval quality,
agreement with exact search, speed, build time, memory and disk use."""
import json
import re
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from . import config
from .chroma_store import ChromaStore
from .embed import Embedder
from .hnsw import HNSW
from .store import VectorStore

METHODS = ("flat", "hnsw", "chroma")
LABELS = {"flat": "Exact search (ours)", "hnsw": "HNSW (ours)", "chroma": "ChromaDB"}


def _relevant(hit: dict, item: dict) -> bool:
    text = re.sub(r"\s+", " ", hit["text"])
    return (any(d in hit["doc"] for d in item["docs"])
            and re.search(item["answer"], text, re.I) is not None)


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t) * 1000


def run(questions_path: str, n_agreement: int = 500, seed: int = 0) -> dict:
    items = json.loads(Path(questions_path).read_text())
    embedder = Embedder(config.MODEL_NAME)
    store = VectorStore.open(config.STORE_DIR)
    chroma = ChromaStore(config.CHROMA_DIR)

    def search(method, q, k):
        if method == "chroma":
            return chroma.search(q, k)
        return store.search(q, k, method=method, ef=config.HNSW_EF_SEARCH)

    embedder.encode([items[0]["question"]])  # warm-up
    qvecs, embed_times = [], []
    for it in items:
        vec, ms = _timed(embedder.encode, [it["question"]])
        qvecs.append(vec[0])
        embed_times.append(ms)
    qvecs = np.stack(qvecs)
    embed_ms = statistics.mean(embed_times)

    for m in METHODS:
        search(m, qvecs[0], 10)  # warm-up

    quality = {m: {"recall5": [], "rr10": [], "latency_ms": []} for m in METHODS}
    per_question = []
    for it, q in zip(items, qvecs):
        row = {"question": it["question"]}
        for m in METHODS:
            hits, ms = _timed(search, m, q, 10)
            ranks = [i for i, h in enumerate(hits, 1) if _relevant(h, it)]
            first = ranks[0] if ranks else None
            quality[m]["recall5"].append(1.0 if first and first <= 5 else 0.0)
            quality[m]["rr10"].append(1.0 / first if first else 0.0)
            quality[m]["latency_ms"].append(ms)
            row[m] = first
            if m == "hnsw":
                row["top_hit"] = f"{hits[0]['title']} (p. {hits[0]['page_start']})"
        per_question.append(row)

    rng = np.random.default_rng(seed)
    live_rows = np.flatnonzero(store.live)
    sample = store.vectors[rng.choice(live_rows, min(n_agreement, len(live_rows)), replace=False)]
    agreement = {"hnsw": [], "chroma": []}
    latency_sample = {m: [] for m in METHODS}
    for q in sample:
        truth, ms = _timed(search, "flat", q, 10)
        latency_sample["flat"].append(ms)
        truth_ids = {h["chunk_id"] for h in truth}
        for m in ("hnsw", "chroma"):
            got, ms = _timed(search, m, q, 10)
            latency_sample[m].append(ms)
            agreement[m].append(len(truth_ids & {h["chunk_id"] for h in got}) / len(truth_ids))

    tracemalloc.start()
    graph = HNSW.load(config.STORE_DIR / "hnsw.json", store.vectors)
    hnsw_traced = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    hnsw_graph_bytes = hnsw_traced - graph.data.nbytes

    build = json.loads((config.DATA / "build_stats.json").read_text())
    results = {
        "n_questions": len(items),
        "n_chunks": len(store),
        "n_agreement_queries": len(sample),
        "query_embedding_ms": embed_ms,
        "methods": {},
        "per_question": per_question,
        "build": build,
        "memory": {
            "vectors_bytes": int(store.vectors.nbytes),
            "hnsw_graph_python_bytes": int(hnsw_graph_bytes),
            "hnsw_edges": graph.n_edges(),
        },
        "disk": {
            "ours_bytes": _dir_size(config.STORE_DIR),
            "ours_without_hnsw_bytes": _dir_size(config.STORE_DIR) - (config.STORE_DIR / "hnsw.json").stat().st_size,
            "chroma_bytes": _dir_size(config.CHROMA_DIR),
        },
    }
    for m in METHODS:
        lat = latency_sample[m]
        results["methods"][m] = {
            "recall_at_5": statistics.mean(quality[m]["recall5"]),
            "mrr_at_10": statistics.mean(quality[m]["rr10"]),
            "agreement_with_exact_at_10": 1.0 if m == "flat" else statistics.mean(agreement[m]),
            "latency_ms_mean": statistics.mean(lat),
            "latency_ms_p95": float(np.percentile(lat, 95)),
        }

    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "evaluation.json").write_text(json.dumps(results, indent=2))
    (config.REPORTS / "evaluation.md").write_text(_markdown(results))
    print(_markdown(results))
    return results


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _markdown(r: dict) -> str:
    m, b = r["methods"], r["build"]
    lines = [
        "# SARS vector database: evaluation",
        "",
        f"{r['n_chunks']:,} chunks from {b['pages']:,} pages, embedded with all-MiniLM-L6-v2 (384 dimensions). "
        f"Both HNSW indexes use 16 links per node, construction width 100 and search width 64.",
        "",
        "## Retrieval quality and speed",
        "",
        f"Quality is measured on {r['n_questions']} hand-written SARS questions. A result counts as correct "
        "when it comes from an expected guide and contains the answer phrase. Agreement and speed are "
        f"measured on {r['n_agreement_queries']} further queries drawn from the stored chunks.",
        "",
        "| Method | Recall@5 | MRR@10 | Agreement with exact top-10 | Mean query time | 95th percentile |",
        "|---|---|---|---|---|---|",
    ]
    for k in METHODS:
        x = m[k]
        lines.append(
            f"| {LABELS[k]} | {x['recall_at_5']:.2f} | {x['mrr_at_10']:.3f} | "
            f"{x['agreement_with_exact_at_10']:.3f} | {x['latency_ms_mean']:.2f} ms | {x['latency_ms_p95']:.2f} ms |"
        )
    lines += [
        "",
        f"Turning one question into a vector takes {r['query_embedding_ms']:.1f} ms on average on this CPU "
        f"(about {r['query_embedding_ms'] / max(x['latency_ms_mean'] for x in m.values()):.0f}× the slowest search), "
        "and is the same for all three methods.",
        "",
        "## Build time, memory and disk",
        "",
        "| Step | Time |",
        "|---|---|",
        f"| Text extraction | {b['extract_s']:.0f} s |",
        f"| Chunking | {b['chunk_s']:.0f} s |",
        f"| Embedding | {b['embed_s']:.0f} s |",
        f"| Exact store (vectors + SQLite) | {b['flat_build_s']:.1f} s |",
        f"| HNSW index (ours, pure Python) | {b['hnsw_build_s']:.0f} s |",
        f"| ChromaDB (HNSW in Rust, plus storage) | {b['chroma_build_s']:.0f} s |",
        "",
        f"- Vector matrix in memory: {_mb(r['memory']['vectors_bytes'])}",
        f"- Our HNSW graph in memory: {_mb(r['memory']['hnsw_graph_python_bytes'])} "
        f"for {r['memory']['hnsw_edges']:,} links (Python lists; about {_mb(r['memory']['hnsw_edges'] * 4)} as 32-bit integers)",
        f"- On disk, ours: {_mb(r['disk']['ours_bytes'])} "
        f"({_mb(r['disk']['ours_without_hnsw_bytes'])} without the HNSW graph file)",
        f"- On disk, ChromaDB: {_mb(r['disk']['chroma_bytes'])}",
        "- ChromaDB's in-memory size isn't reported: its index lives in native code that Python's memory tracing can't see.",
        "",
        "## Per question (rank of first correct result, top 10)",
        "",
        "| # | Question | Exact | HNSW | ChromaDB | Top HNSW result |",
        "|---|---|---|---|---|---|",
    ]
    for i, q in enumerate(r["per_question"], 1):
        cells = [str(q[k]) if q[k] else "–" for k in METHODS]
        lines.append(f"| {i} | {q['question']} | {' | '.join(cells)} | {q['top_hit']} |")
    return "\n".join(lines) + "\n"
