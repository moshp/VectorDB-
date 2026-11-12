"""Static matplotlib figures: a guide-similarity network and a 3D t-SNE scatter."""
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch

from . import config
from .store import VectorStore

# The reference figures' soft palette first, then further shades for topics beyond nine.
COLORS = [
    "#ff4f4f", "#f2b8b8", "#5fb3c9", "#86e693", "#4f8ff7", "#b9b9f2", "#c8f29a", "#f5f0b3", "#c9f5e8",
    "#ffb26b", "#d6a77a", "#c4c4c4", "#c79be8", "#c9c96b", "#6fd9e8", "#ff9e85", "#7d92d9", "#f290d2",
    "#a6e25a", "#4fb5aa",
]


def short_label(label: str, n: int = 2) -> str:
    """'dividend, dividends, company, tax' -> 'Dividend, company'."""
    kept = []
    for term in label.split(", "):
        if not any(term.startswith(k) or k.startswith(term) for k in kept):
            kept.append(term)
    return ", ".join(kept[:n]).capitalize()


def _load():
    data = json.loads((config.DATA / "plot_data.json").read_text())
    store = VectorStore.open(config.STORE_DIR)
    chunks = store.live_chunks()
    if len(chunks) != len(data["points"]):
        raise RuntimeError("plot_data.json is out of date; run `python3 -m sarsvdb plot` first")
    vectors = store.vectors[[c["row"] for c in chunks]]
    topics = np.array([p[3] for p in data["points"]])
    return data, chunks, vectors, topics


def _ring_layout(graph: nx.Graph, gap: float = 1.4) -> dict:
    """Lays out each connected group on its own, then places the groups
    around a ring, largest first."""
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    parts = []
    for comp in comps:
        nodes = sorted(comp)
        radius = 0.95 * np.sqrt(len(nodes))
        if len(nodes) == 1:
            local = {nodes[0]: np.zeros(2)}
        else:
            # Spacing by link count keeps circles apart; exact similarities would stack
            # near-duplicate guides on top of each other.
            local = nx.kamada_kawai_layout(graph.subgraph(nodes), weight=None, scale=radius)
        parts.append((nodes, local, radius))
    circumference = sum(2 * r + gap for _, _, r in parts)
    ring = circumference / (2 * np.pi)
    pos, angle = {}, np.pi / 2
    for nodes, local, radius in parts:
        step = (2 * radius + gap) / ring
        angle -= step / 2
        centre = ring * np.array([np.cos(angle), np.sin(angle)])
        for n in nodes:
            pos[n] = centre + local[n]
        angle -= step / 2
    return pos


def network_figure(data, chunks, vectors, topics, neighbours: int = 1) -> list[dict]:
    """One node per guide, linked to its most similar guide (cosine
    similarity of the guides' average passage vectors). Chains of nearest
    neighbours form separate groups, laid out around a ring."""
    docs = sorted({c["doc"] for c in chunks})
    rows = {d: [i for i, c in enumerate(chunks) if c["doc"] == d] for d in docs}
    guides = []
    for d in docs:
        idx = rows[d]
        v = vectors[idx].mean(axis=0)
        guides.append({
            "doc": d,
            "title": chunks[idx[0]]["title"],
            "vector": v / np.linalg.norm(v),
            "passages": len(idx),
            "topic": Counter(topics[idx].tolist()).most_common(1)[0][0],
        })
    guides.sort(key=lambda g: (g["topic"], -g["passages"]))
    for n, g in enumerate(guides, 1):
        g["number"] = n

    V = np.stack([g["vector"] for g in guides])
    dist = np.clip(1.0 - V @ V.T, 0.0, None)
    np.fill_diagonal(dist, np.inf)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(guides)))
    for i in range(len(guides)):
        for j in np.argsort(dist[i])[:neighbours]:
            graph.add_edge(i, int(j))

    pos = _ring_layout(graph)
    sizes = np.array([g["passages"] for g in guides], dtype=float)
    area = 110 + 850 * (np.sqrt(sizes) - np.sqrt(sizes.min())) / (np.sqrt(sizes.max()) - np.sqrt(sizes.min()))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#ffd08f", width=1.1)
    order = np.argsort(-area)  # small circles drawn on top of large ones
    xy = np.array([pos[i] for i in order])
    ax.scatter(xy[:, 0], xy[:, 1], s=area[order], c=[COLORS[guides[i]["topic"] % len(COLORS)] for i in order],
               edgecolors="#8a8a8a", linewidths=0.8, alpha=0.95, zorder=2)
    for i in order:
        ax.text(*pos[i], str(guides[i]["number"]), ha="center", va="center", fontsize=6.5, zorder=3)

    used = sorted({g["topic"] for g in guides})
    handles = [Patch(facecolor=COLORS[t % len(COLORS)], edgecolor="#8a8a8a", linewidth=0.5,
                     label=short_label(data["topics"][t]["label"])) for t in used]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True, handlelength=2.4, handleheight=0.9)
    ax.set_axis_off()
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    span = max(np.ptp(xs), np.ptp(ys))
    pad = 0.06 * span
    # Extra room on the left keeps the legend clear of the circles.
    ax.set_xlim(xs.min() - pad - 0.22 * span, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    fig.tight_layout()
    config.PLOTS.mkdir(exist_ok=True)
    fig.savefig(config.PLOTS / "guide_network.png", facecolor="white")
    plt.close(fig)
    return guides


def tsne3d_figure(data, topics) -> None:
    xyz = np.array([p[:3] for p in data["points"]])
    fig = plt.figure(figsize=(7.2, 6.6), dpi=150)
    ax = fig.add_subplot(projection="3d")
    for t in range(data["k"]):
        m = topics == t
        ax.scatter(xyz[m, 0], xyz[m, 1], xyz[m, 2], s=8, color=COLORS[t % len(COLORS)], alpha=0.9,
                   linewidths=0, depthshade=False, label=short_label(data["topics"][t]["label"]))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.93, 0.93, 0.93, 1.0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 1)
        axis._axinfo["grid"]["linewidth"] = 0.8
    ax.set_xlabel("V1")
    ax.set_ylabel("V2")
    ax.set_zlabel("V3")
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("Clustering 3D plot based on T-SNE scores", fontsize=11)
    ax.legend(loc="upper left", fontsize=6, markerscale=1.6, handletextpad=0.2, borderpad=0.5,
              labelspacing=0.3, frameon=True, framealpha=0.9, bbox_to_anchor=(-0.08, 1.0))
    fig.tight_layout()
    config.PLOTS.mkdir(exist_ok=True)
    fig.savefig(config.PLOTS / "tsne_3d.png", facecolor="white")
    plt.close(fig)


def write_key(data, guides) -> None:
    lines = [
        "# Guide network key",
        "",
        "Numbers in `guide_network.png`. Colour is the topic most of the guide's passages fall into; "
        "circle size is the number of passages.",
        "",
        "| # | Guide | Topic | Passages | PDF |",
        "|---|---|---|---|---|",
    ]
    for g in guides:
        lines.append(f"| {g['number']} | {g['title']} | {short_label(data['topics'][g['topic']]['label'])} "
                     f"| {g['passages']} | `{g['doc']}` |")
    (config.PLOTS / "guide_network_key.md").write_text("\n".join(lines) + "\n")


def run() -> None:
    data, chunks, vectors, topics = _load()
    guides = network_figure(data, chunks, vectors, topics)
    write_key(data, guides)
    tsne3d_figure(data, topics)
    print(f"Wrote {config.PLOTS / 'guide_network.png'}, {config.PLOTS / 'guide_network_key.md'} "
          f"and {config.PLOTS / 'tsne_3d.png'}")
