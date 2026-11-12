"""Part 2, retrieval + generation: embed question -> HNSW top-k -> re-rank ->
build prompt -> Llama 3.2 3B (llama.cpp) -> answer with citations."""
import os
import re
import threading
import time
from typing import Optional

from . import config
from .pipeline import Ingestor

NO_ANSWER = "I don't know based on the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered passages you are given.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Prefer the general rule. A worked example applies only to the case it describes; if you use one, say which case it is.
- Put the passage number in square brackets after each fact you use, for example: The rate is 15% [2].
- If the passages do not contain the answer, reply with exactly this sentence and nothing else: {NO_ANSWER}
- Answer in at most four sentences."""

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class Generator:
    def __init__(self, model_path=config.LLM_PATH):
        from llama_cpp import Llama

        if not model_path.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(config.LLM_REPO, config.LLM_FILE, local_dir=model_path.parent)
        self.llm = Llama(model_path=str(model_path), n_ctx=config.LLM_CONTEXT,
                         n_threads=os.cpu_count(), verbose=False)
        self.lock = threading.Lock()  # one generation at a time

    def generate(self, question: str, passages: list[dict]) -> tuple[str, dict]:
        blocks = []
        for n, p in enumerate(passages, start=1):
            pages = f"p. {p['page_start']}" if p["page_start"] == p["page_end"] else \
                f"pp. {p['page_start']}-{p['page_end']}"
            blocks.append(f"[{n}] ({p['title']}, {pages})\n{p['text']}")
        user = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
        with self.lock:
            out = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                max_tokens=config.LLM_MAX_ANSWER_TOKENS, temperature=0.0,
            )
        return out["choices"][0]["message"]["content"].strip(), out["usage"]


class Retriever:
    def __init__(self, ingestor: Ingestor, use_reranker: bool = True):
        self.ingestor = ingestor
        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")

    def retrieve(self, question: str, k: int = config.RETRIEVE_K, keep: int = config.CONTEXT_K,
                 rerank: bool = True, docs: Optional[list[str]] = None) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        q = self.ingestor.embedder.encode_queries([question])[0]
        with self.ingestor.lock:
            hits = self.ingestor.store.search(q, k, docs=docs, method="hnsw", ef=config.HNSW_EF_SEARCH)
        t1 = time.perf_counter()
        if rerank and self.reranker is not None and hits:
            scores = self.reranker.predict([(question, h["text"]) for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda h: -h["rerank_score"])
        t2 = time.perf_counter()
        return hits[:keep], {"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000}


def parse_citations(answer: str, n_passages: int) -> list[int]:
    cited = []
    for group in CITATION.findall(answer):
        for n in group.split(","):
            n = int(n)
            if 1 <= n <= n_passages and n not in cited:
                cited.append(n)
    return cited


class RAG:
    def __init__(self, ingestor: Ingestor, retriever: Retriever, generator: Generator):
        self.ingestor = ingestor
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str, rerank: bool = True, docs: Optional[list[str]] = None) -> dict:
        passages, timings = self.retriever.retrieve(question, rerank=rerank, docs=docs)
        if not passages:
            return {"answer": "There are no documents to search yet. Upload a document first.",
                    "abstained": True, "citations": [], "passages": [], "timings": timings}
        t = time.perf_counter()
        text, usage = self.generator.generate(question, passages)
        timings["generate_ms"] = (time.perf_counter() - t) * 1000
        abstained = NO_ANSWER.lower().rstrip(".") in text.lower()
        cited = [] if abstained else parse_citations(text, len(passages))
        fields = ("doc", "title", "page_start", "page_end", "text", "score")
        return {
            "answer": text,
            "abstained": abstained,
            "citations": cited,
            "passages": [{"n": n, **{f: p[f] for f in fields}, "rerank_score": p.get("rerank_score")}
                         for n, p in enumerate(passages, start=1)],
            "usage": {"prompt_tokens": usage["prompt_tokens"], "answer_tokens": usage["completion_tokens"]},
            "timings": timings,
        }
