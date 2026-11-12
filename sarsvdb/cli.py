import argparse
import json
import shutil
import textwrap
import time

from . import config


def _load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cmd_build(args) -> None:
    from .chroma_store import ChromaStore
    from .chunking import chunk_pages
    from .embed import Embedder
    from .ingest import ingest
    from .store import VectorStore

    stats = {}
    t = time.perf_counter()
    print("1/5 Extracting text from PDFs ...", flush=True)
    stats["pages"] = ingest(config.MANIFEST, config.PDF_DIR, config.PAGES_FILE)
    stats["extract_s"] = time.perf_counter() - t

    embedder = Embedder(config.MODEL_NAME)
    t = time.perf_counter()
    print("2/5 Chunking ...", flush=True)
    chunks = chunk_pages(_load_jsonl(config.PAGES_FILE), embedder.tokenizer,
                         config.CHUNK_TOKENS, config.OVERLAP_TOKENS, config.MAX_CHUNK_TOKENS)
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats["chunks"] = len(chunks)
    stats["chunk_s"] = time.perf_counter() - t
    print(f"    {len(chunks)} chunks", flush=True)

    t = time.perf_counter()
    print("3/5 Embedding ...", flush=True)
    vectors = embedder.encode([c["text"] for c in chunks], batch_size=args.batch_size, progress=True)
    stats["embed_s"] = time.perf_counter() - t

    print("4/5 Building our vector store (exact + HNSW) ...", flush=True)
    shutil.rmtree(config.STORE_DIR, ignore_errors=True)
    store = VectorStore(config.STORE_DIR, embedder.dim, config.MODEL_NAME)
    t = time.perf_counter()
    store.add(chunks, vectors)
    stats["flat_build_s"] = time.perf_counter() - t
    t = time.perf_counter()
    store.build_hnsw(config.HNSW_M, config.HNSW_EF_CONSTRUCTION, progress=True)
    stats["hnsw_build_s"] = time.perf_counter() - t
    store.save()

    print("5/5 Building ChromaDB ...", flush=True)
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    t = time.perf_counter()
    ChromaStore(config.CHROMA_DIR).build(chunks, vectors, config.HNSW_M,
                                         config.HNSW_EF_CONSTRUCTION, config.HNSW_EF_SEARCH)
    stats["chroma_build_s"] = time.perf_counter() - t

    (config.DATA / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def cmd_query(args) -> None:
    from .embed import Embedder
    embedder = Embedder(config.MODEL_NAME)
    q = embedder.encode([args.question])[0]
    docs = [args.doc] if args.doc else None
    if args.method == "chroma":
        from .chroma_store import ChromaStore
        hits = ChromaStore(config.CHROMA_DIR).search(q, args.k, docs)
    else:
        from .store import VectorStore
        hits = VectorStore.open(config.STORE_DIR).search(q, args.k, docs, method=args.method,
                                                        ef=config.HNSW_EF_SEARCH)
    for i, h in enumerate(hits, 1):
        pages = (f"p. {h['page_start']}" if h["page_start"] == h["page_end"]
                 else f"pp. {h['page_start']}-{h['page_end']}")
        print(f"\n[{i}] score {h['score']:.3f} | {h['title']} ({pages})")
        print(textwrap.indent(textwrap.shorten(h["text"].replace("\n", " "), 400), "    "))


def cmd_eval(args) -> None:
    from .evaluate import run
    run(args.questions)


def cmd_plot(args) -> None:
    from .plot import run
    run(args.k_min, args.k_max, args.perplexity)


def main() -> None:
    ap = argparse.ArgumentParser(prog="sarsvdb", description="SARS vector database")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="extract, chunk, embed and index all PDFs")
    b.add_argument("--batch-size", type=int, default=32)
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="semantic search")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--method", choices=["flat", "hnsw", "chroma"], default="hnsw")
    q.add_argument("--doc", help="only search this PDF file name")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="compare our store and ChromaDB")
    e.add_argument("--questions", default=str(config.DATA / "eval_questions.json"))
    e.set_defaults(func=cmd_eval)

    p = sub.add_parser("plot", help="topic clustering and 3D t-SNE data")
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=25)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
