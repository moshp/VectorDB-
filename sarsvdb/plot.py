"""Topic discovery (k-means, number of topics chosen by silhouette score) and
3D t-SNE coordinates for the interactive plot."""
import json
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from . import config
from .store import VectorStore

SNIPPET_CHARS = 220


def choose_k(vectors: np.ndarray, k_min: int, k_max: int, seed: int = 42) -> tuple[int, dict, np.ndarray]:
    scores, labels_by_k = {}, {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(vectors)
        scores[k] = float(silhouette_score(vectors, labels, metric="cosine",
                                           sample_size=min(4000, len(vectors)), random_state=seed))
        labels_by_k[k] = labels
        print(f"  k={k:2d}  silhouette={scores[k]:.4f}", flush=True)
    best = max(scores, key=scores.get)
    return best, scores, labels_by_k[best]


# Words found throughout every SARS guide; they say nothing about a topic.
DOMAIN_STOP_WORDS = {
    "sars", "tax", "taxes", "section", "sections", "guide", "act", "page", "pages", "revision",
    "external", "paragraph", "paragraphs", "chapter", "example", "examples", "refer", "note",
    "also", "must", "may", "will", "shall", "however", "applicable", "relevant", "terms",
    "south", "africa", "african", "respect", "regard", "following", "person", "persons",
}
NAMES_FILE = config.DATA / "topic_names.json"


def topic_terms(texts: list[str], labels: np.ndarray, k: int, n_terms: int = 5) -> list[list[str]]:
    """Class-based TF-IDF over words and two-word phrases: terms frequent in
    one topic but rare across topics. A word is dropped when a phrase
    containing it already made the list."""
    stop = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)
    vec = CountVectorizer(stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
                          ngram_range=(1, 2), min_df=5)
    counts = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    per_topic = np.vstack([np.asarray(counts[labels == t].sum(axis=0)).ravel() for t in range(k)])
    tf = per_topic / per_topic.sum(axis=1, keepdims=True)
    idf = np.log(1 + per_topic.sum(axis=1).mean() / per_topic.sum(axis=0))
    scores = tf * idf
    out = []
    for t in range(k):
        picked: list[str] = []
        for term in vocab[np.argsort(-scores[t])[:60]]:
            words = set(term.split())
            if any(words <= set(p.split()) or set(p.split()) <= words for p in picked):
                continue
            picked.append(str(term))
            if len(picked) == n_terms:
                break
        out.append(picked)
    return out


def representative(vectors: np.ndarray, labels: np.ndarray, chunks: list[dict], t: int, n: int = 4) -> list[str]:
    """The passages closest to the topic's centre."""
    idx = np.flatnonzero(labels == t)
    centre = vectors[idx].mean(axis=0)
    best = idx[np.argsort(-(vectors[idx] @ centre))[:n]]
    return [f"{chunks[i]['title']} (p. {chunks[i]['page_start']}): " + " ".join(chunks[i]["text"].split())[:300]
            for i in best]


def topic_names(topics: list[dict]) -> list[str]:
    """Human-written topic names from data/topic_names.json. Each entry keeps
    the terms it was written for; if the clustering has changed since, the
    terms no longer match and the topic falls back to its terms."""
    saved = json.loads(NAMES_FILE.read_text()) if NAMES_FILE.exists() else []
    names = []
    for tp in topics:
        match = next((s["name"] for s in saved if s["terms"] == tp["terms"]), None)
        if match is None:
            print(f"  no saved name for topic {tp['id'] + 1} ({', '.join(tp['terms'])}); using its terms")
        names.append(match or ", ".join(tp["terms"][:3]).capitalize())
    return names


def run(k_min: int, k_max: int, perplexity: float, seed: int = 42) -> None:
    store = VectorStore.open(config.EVAL_STORE)
    chunks = store.live_chunks()
    vectors = store.vectors[[c["row"] for c in chunks]]

    print(f"Choosing the number of topics between {k_min} and {k_max} ...", flush=True)
    k, sil, labels = choose_k(vectors, k_min, k_max, seed)
    terms = topic_terms([c["text"] for c in chunks], labels, k)
    print(f"Chose k={k}", flush=True)

    print("Running 3D t-SNE ...", flush=True)
    t = time.perf_counter()
    xyz = TSNE(n_components=3, perplexity=perplexity, init="pca", metric="cosine",
               random_state=seed).fit_transform(vectors)
    tsne_s = time.perf_counter() - t
    print(f"  t-SNE took {tsne_s:.0f} s", flush=True)

    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = {int(old): new for new, old in enumerate(order)}
    topics = []
    for new, old in enumerate(order):
        members = [c for c, l in zip(chunks, labels) if l == old]
        doc_counts = {}
        for c in members:
            doc_counts[c["title"]] = doc_counts.get(c["title"], 0) + 1
        topics.append({
            "id": new,
            "terms": terms[old],
            "size": len(members),
            "top_docs": sorted(doc_counts.items(), key=lambda kv: -kv[1])[:3],
            "examples": representative(vectors, labels, chunks, old),
        })
    for tp, name in zip(topics, topic_names(topics)):
        tp["label"] = name

    points = [
        [round(float(x), 2), round(float(y), 2), round(float(z), 2), remap[int(l)],
         c["title"], c["page_start"], " ".join(c["text"].split())[:SNIPPET_CHARS]]
        for (x, y, z), l, c in zip(xyz, labels, chunks)
    ]
    out = {
        "model": config.MODEL_NAME,
        "dim": int(vectors.shape[1]),
        "k": k,
        "silhouette": sil,
        "perplexity": perplexity,
        "tsne_seconds": tsne_s,
        "n_points": len(points),
        "n_docs": len({c["doc"] for c in chunks}),
        "topics": topics,
        "points": points,
    }
    (config.DATA / "plot_data.json").write_text(json.dumps(out, ensure_ascii=False))
    for tp in topics:
        print(f"  topic {tp['id'] + 1:2d} ({tp['size']:4d} chunks): {tp['label']}  [{', '.join(tp['terms'])}]")
    render()


def rename() -> None:
    """Re-applies data/topic_names.json to the saved plot data without re-clustering."""
    path = config.DATA / "plot_data.json"
    data = json.loads(path.read_text())
    for tp, name in zip(data["topics"], topic_names(data["topics"])):
        tp["label"] = name
    path.write_text(json.dumps(data, ensure_ascii=False))
    render()


def render() -> None:
    """Fills the topic map page template with plot_data.json."""
    data = (config.DATA / "plot_data.json").read_text()
    template = (config.ROOT / "viz" / "topic_map_template.html").read_text()
    safe = data.replace("</", "<\\/")
    config.REPORTS.mkdir(exist_ok=True)
    out = config.REPORTS / "topic_map.html"
    out.write_text(template.replace("/*__DATA__*/null", safe, 1), encoding="utf-8")
    print(f"Wrote {out}")
