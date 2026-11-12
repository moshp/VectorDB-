"""Part 3, evaluation: ingests the SARS guides into a separate test database
through the same pipeline as the web app, then measures retrieval (with and
without re-ranking), answers against reference answers, speed and OCR."""
import difflib
import json
import random
import re
import shutil
import statistics
import time
from pathlib import Path

from pypdf import PdfReader

from . import config
from .embed import Embedder
from .ocr import ocr_pdf_page
from .pipeline import Ingestor
from .rag import CITATION, RAG, Generator, Retriever, load_reranker

STATS_FILE = config.EVAL_STORE / "ingest_stats.json"


def build_eval_store(rebuild: bool = False) -> None:
    if rebuild:
        shutil.rmtree(config.EVAL_STORE, ignore_errors=True)
    manifest = json.loads(config.MANIFEST.read_text())
    names = sorted(n for n, m in manifest.items() if m.get("usable", True))
    ingestor = Ingestor(config.EVAL_STORE, Embedder(config.MODEL_NAME, config.QUERY_PREFIX))
    done = {d["doc"] for d in ingestor.documents()}
    stats = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    for n, name in enumerate(names, 1):
        if name in done:
            continue
        t = time.perf_counter()
        result = ingestor.ingest(config.PDF_DIR / name)
        result["total_s"] = time.perf_counter() - t
        stats[name] = result
        STATS_FILE.write_text(json.dumps(stats, indent=2))
        extra = f"{result.get('chunks', 0)} chunks, OCR pages {result.get('ocr_pages', [])}" \
            if result["status"] == "added" else result["status"]
        print(f"[{n}/{len(names)}] {name}: {extra} ({result['total_s']:.1f} s)", flush=True)
    print(f"Test database: {len(ingestor.documents())} documents, {len(ingestor.store)} chunks")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _relevant(passage: dict, item: dict) -> bool:
    return (any(d in passage["doc"] for d in item["docs"])
            and re.search(item["answer"], _norm(passage["text"]), re.I) is not None)


def _first_relevant(passages: list[dict], item: dict):
    return next((i for i, p in enumerate(passages, 1) if _relevant(p, item)), None)


NUMBER = re.compile(r"\d[\d ,.]*\d|\d")


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[ ,]", "", m).rstrip(".") for m in NUMBER.findall(text)} - {""}


def _ocr_check(n_pages: int = 10, seed: int = 7) -> dict:
    """OCR pages that also have a text layer and compare the words."""
    rng = random.Random(seed)
    manifest = json.loads(config.MANIFEST.read_text())
    names = sorted(n for n, m in manifest.items() if m.get("usable", True))
    rows = []
    for name in rng.sample(names, n_pages):
        reader = PdfReader(str(config.PDF_DIR / name))
        page_no = rng.randint(1, len(reader.pages))
        truth = (reader.pages[page_no - 1].extract_text() or "").lower().split()
        if len(truth) < 50:
            continue
        t = time.perf_counter()
        ocr = ocr_pdf_page(config.PDF_DIR / name, page_no).lower().split()
        secs = time.perf_counter() - t
        matcher = difflib.SequenceMatcher(None, truth, ocr, autojunk=False)
        matched = sum(b.size for b in matcher.get_matching_blocks())
        rows.append({"doc": name, "page": page_no, "words": len(truth),
                     "word_recall": matched / len(truth), "seconds": secs})
    return {"pages": rows,
            "mean_word_recall": statistics.mean(r["word_recall"] for r in rows),
            "mean_seconds": statistics.mean(r["seconds"] for r in rows)}


def run(questions_path: str, limit: int | None = None) -> dict:
    items = json.loads(Path(questions_path).read_text())
    if limit:
        items = items[:limit]
    ingestor = Ingestor(config.EVAL_STORE, Embedder(config.MODEL_NAME, config.QUERY_PREFIX))
    retriever = Retriever(ingestor, load_reranker())
    rag = RAG(ingestor, retriever, Generator())

    rows = []
    for n, it in enumerate(items, 1):
        answerable = it.get("answerable", True)
        row = {"question": it["question"], "answerable": answerable}
        if answerable:
            plain, _ = retriever.retrieve(it["question"], keep=config.CONTEXT_K, rerank=False)
            row["rank_without_rerank"] = _first_relevant(plain, it)
        r = rag.answer(it["question"])
        row.update(answer=r["answer"], abstained=r["abstained"], citations=r["citations"],
                   timings=r["timings"])
        if answerable:
            row["rank_with_rerank"] = _first_relevant(r["passages"], it)
            row["facts_found"] = [bool(re.search(k, _norm(r["answer"]), re.I)) for k in it["key_facts"]]
            row["correct"] = not r["abstained"] and all(row["facts_found"])
            cited = [r["passages"][c - 1] for c in r["citations"]]
            row["cites_relevant"] = any(_relevant(p, it) for p in cited)
            cited_text = " ".join(p["text"] for p in cited)
            answer_text = CITATION.sub(" ", r["answer"])
            row["unsupported_numbers"] = sorted(_numbers(answer_text) - _numbers(cited_text))
        else:
            row["correct"] = r["abstained"]
        rows.append(row)
        status = "ok" if row["correct"] else "WRONG"
        print(f"[{n}/{len(items)}] {status}: {it['question']}\n      -> {r['answer']}", flush=True)

    print("Checking OCR accuracy ...", flush=True)
    ocr = _ocr_check()
    ingest = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    results = {"model": config.MODEL_NAME, "llm": config.LLM_FILE, "reranker": config.RERANK_MODEL,
               "questions": rows, "ocr": ocr, "ingest": ingest,
               "n_documents": len(ingestor.documents()), "n_chunks": len(ingestor.store)}
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "rag_evaluation.json").write_text(json.dumps(results, indent=2))
    report = _markdown(results)
    (config.REPORTS / "rag_evaluation.md").write_text(report)
    print(report)
    return results


def rescore(questions_path: str) -> dict:
    """Re-applies the key facts in the question file to the saved answers,
    without running retrieval or the model again."""
    path = config.REPORTS / "rag_evaluation.json"
    results = json.loads(path.read_text())
    items = {it["question"]: it for it in json.loads(Path(questions_path).read_text())}
    for row in results["questions"]:
        it = items[row["question"]]
        if row["answerable"]:
            row["facts_found"] = [bool(re.search(k, _norm(row["answer"]), re.I)) for k in it["key_facts"]]
            row["correct"] = not row["abstained"] and all(row["facts_found"])
    path.write_text(json.dumps(results, indent=2))
    report = _markdown(results)
    (config.REPORTS / "rag_evaluation.md").write_text(report)
    print(report)
    return results


def _pct(xs) -> str:
    xs = list(xs)
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "–"


def _markdown(r: dict) -> str:
    qs = r["questions"]
    ans = [q for q in qs if q["answerable"]]
    neg = [q for q in qs if not q["answerable"]]
    t = {k: statistics.mean(q["timings"].get(k, 0) for q in qs) / 1000
         for k in ("retrieve_ms", "rerank_ms", "generate_ms")}
    added = [s for s in r["ingest"].values() if s.get("status") == "added"]
    lines = [
        "# Document Q&A: evaluation",
        "",
        f"Test database: {r['n_documents']} SARS guides, {r['n_chunks']:,} chunks, ingested through the "
        f"same pipeline as the web app. Embeddings: {r['model'].split('/')[-1]}; re-ranker: "
        f"{r['reranker'].split('/')[-1]}; answers: {r['llm']} (llama.cpp, CPU).",
        "",
        "## Retrieval",
        "",
        f"For each of the {len(ans)} answerable questions, is a passage that contains the answer among the "
        f"{config.CONTEXT_K} passages given to the model?",
        "",
        "| | Correct passage in top 5 | Mean reciprocal rank |",
        "|---|---|---|",
    ]
    for label, key in (("HNSW only", "rank_without_rerank"),
                       (f"HNSW top {config.RETRIEVE_K}, re-ranked to 5", "rank_with_rerank")):
        ranks = [q[key] for q in ans]
        mrr = statistics.mean(1 / k if k else 0 for k in ranks)
        lines.append(f"| {label} | {_pct(k is not None for k in ranks)} | {mrr:.3f} |")
    lines += [
        "",
        "## Answers",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Answerable questions answered correctly (every key fact present) | {_pct(q['correct'] for q in ans)} |",
        f"| Answerable questions where the model said it didn't know | {_pct(q['abstained'] for q in ans)} |",
        f"| Answers citing a passage that contains the answer | "
        f"{_pct(q['cites_relevant'] for q in ans if not q['abstained'])} |",
        f"| Answers whose numbers all appear in the cited passages | "
        f"{_pct(not q['unsupported_numbers'] for q in ans if not q['abstained'])} |",
        f"| Unanswerable questions where the model said it didn't know | {_pct(q['correct'] for q in neg)} |",
        "",
        "## Speed (per question, CPU)",
        "",
        f"- Search: {t['retrieve_ms'] * 1000:.0f} ms",
        f"- Re-ranking 20 passages: {t['rerank_ms'] * 1000:.0f} ms",
        f"- Writing the answer: {t['generate_ms']:.1f} s",
        "",
        "## Ingestion and OCR",
        "",
    ]
    if added:
        pages = sum(s["pages"] for s in added)
        total = sum(s["total_s"] for s in added)
        ocr_pages = sum(len(s["ocr_pages"]) for s in added)
        lines.append(f"- {len(added)} documents, {pages:,} pages ingested in {total / 60:.1f} min "
                     f"({total / pages:.2f} s per page); {ocr_pages} pages needed OCR.")
    o = r["ocr"]
    lines += [
        f"- OCR check: {len(o['pages'])} pages that also have a text layer were OCR'd at 300 dpi; "
        f"on average {100 * o['mean_word_recall']:.1f}% of the real words were recovered in order, "
        f"at {o['mean_seconds']:.1f} s per page.",
        "",
        "## Every question",
        "",
        "| # | Question | Correct | Rank (HNSW → re-ranked) | Answer |",
        "|---|---|---|---|---|",
    ]
    for i, q in enumerate(qs, 1):
        ranks = f"{q.get('rank_without_rerank') or '–'} → {q.get('rank_with_rerank') or '–'}" \
            if q["answerable"] else "(not in the documents)"
        answer = q["answer"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {q['question']} | {'yes' if q['correct'] else 'no'} | {ranks} | {answer} |")
    return "\n".join(lines) + "\n"
