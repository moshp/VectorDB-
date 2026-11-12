import argparse

from . import config


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("sarsvdb.app:app", host=args.host, port=args.port, log_level="info")


def cmd_ingest_eval(args) -> None:
    from .rag_eval import build_eval_store
    build_eval_store(rebuild=args.rebuild)


def cmd_eval(args) -> None:
    from .rag_eval import rescore, run
    if args.rescore:
        rescore(args.questions)
    else:
        run(args.questions, limit=args.limit)


def cmd_eval_index(args) -> None:
    from .evaluate import run
    run(args.questions)


def cmd_plot(args) -> None:
    from .plot import rename, run
    if args.rename:
        rename()
    else:
        run(args.k_min, args.k_max, args.perplexity)


def cmd_figures(args) -> None:
    from .figures import run
    run()


def main() -> None:
    ap = argparse.ArgumentParser(prog="sarsvdb", description="Document Q&A: ingestion, RAG and evaluation")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the web app (upload documents, ask questions)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    i = sub.add_parser("ingest-eval", help="ingest the SARS guides in data/pdfs into the test database")
    i.add_argument("--rebuild", action="store_true", help="delete the test database first")
    i.set_defaults(func=cmd_ingest_eval)

    e = sub.add_parser("eval", help="Part 3: evaluate retrieval, re-ranking, answers and OCR")
    e.add_argument("--questions", default=str(config.DATA / "eval_questions.json"))
    e.add_argument("--limit", type=int, help="only the first N questions (quick check)")
    e.add_argument("--rescore", action="store_true",
                   help="re-score the saved answers with the current key facts, without re-running the model")
    e.set_defaults(func=cmd_eval)

    x = sub.add_parser("eval-index", help="compare exact search, our HNSW and ChromaDB on the test database")
    x.add_argument("--questions", default=str(config.DATA / "eval_questions.json"))
    x.set_defaults(func=cmd_eval_index)

    p = sub.add_parser("plot", help="topic clustering and the interactive 3D t-SNE map (test database)")
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=25)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--rename", action="store_true",
                   help="only re-apply data/topic_names.json to the existing plot data")
    p.set_defaults(func=cmd_plot)

    f = sub.add_parser("figures", help="static guide network and 3D t-SNE images (test database)")
    f.set_defaults(func=cmd_figures)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
