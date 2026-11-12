"""Part 1, ingestion: extract (OCR where needed) -> clean -> chunk -> embed -> store."""
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import config
from .chunking import chunk_document
from .embed import Embedder
from .ingest import extract_document, fingerprint
from .store import VectorStore

Progress = Callable[[str, float], None]


def safe_name(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "document"


class Ingestor:
    """Owns one vector store. All reads and writes go through its lock, so the
    web app can ingest in the background while answering questions."""

    def __init__(self, store_dir: Path, embedder: Embedder, uploads_dir: Optional[Path] = None):
        self.embedder = embedder
        self.store = VectorStore.open_or_create(store_dir, embedder.dim, config.MODEL_NAME)
        self.uploads_dir = uploads_dir
        self.lock = threading.RLock()

    def _unique_doc_name(self, name: str) -> str:
        stem, suffix = Path(name).stem, Path(name).suffix
        candidate, n = name, 2
        while self.store.has_doc_name(candidate):
            candidate = f"{stem}-{n}{suffix}"
            n += 1
        return candidate

    def ingest(self, path: Path, filename: Optional[str] = None, title: Optional[str] = None,
               progress: Optional[Progress] = None) -> dict:
        """Returns {'status': 'added' | 'duplicate' | 'empty', ...}."""
        report = progress or (lambda stage, frac: None)
        filename = safe_name(filename or path.name)

        def on_page(i: int, total: int, method: str) -> None:
            label = "Reading text" if method == "text" else "Running OCR"
            report(f"{label}: page {i} of {total}", 0.6 * i / total)

        report("Reading document", 0.0)
        t0 = time.perf_counter()
        extracted = extract_document(path, on_page)
        t1 = time.perf_counter()
        fp = fingerprint(extracted.pages)
        with self.lock:
            existing = self.store.find_document(fp)
        if existing:
            return {"status": "duplicate", "doc": existing["doc"], "title": existing["title"]}

        with self.lock:
            doc = self._unique_doc_name(filename)
        doc_title = title or extracted.title
        pages = [{"doc": doc, "title": doc_title, "category": None, "url": None, "page": i, "text": t}
                 for i, t in enumerate(extracted.pages, start=1) if len(t.strip()) >= 20]
        if not pages:
            return {"status": "empty", "doc": doc, "title": doc_title,
                    "message": "No readable text was found, even with OCR."}

        report("Splitting into chunks", 0.65)
        chunks = chunk_document(pages, self.embedder.tokenizer, config.CHUNK_TOKENS,
                                config.OVERLAP_TOKENS, config.MAX_CHUNK_TOKENS)
        t2 = time.perf_counter()
        report(f"Embedding {len(chunks)} chunks", 0.7)
        vectors = self.embedder.encode([c["text"] for c in chunks])
        t3 = time.perf_counter()

        report("Saving to the vector database", 0.9)
        with self.lock:
            if self.store.find_document(fp):  # uploaded twice at the same time
                return {"status": "duplicate", "doc": doc, "title": doc_title}
            self.store.add_document(doc, doc_title, fp, len(extracted.pages), extracted.ocr_pages,
                                    chunks, vectors, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        t4 = time.perf_counter()
        if self.uploads_dir:
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, self.uploads_dir / doc)
        report("Done", 1.0)
        return {"status": "added", "doc": doc, "title": doc_title, "pages": len(extracted.pages),
                "ocr_pages": extracted.ocr_pages, "chunks": len(chunks),
                "timings_s": {"extract": t1 - t0, "chunk": t2 - t1, "embed": t3 - t2, "store": t4 - t3}}

    def delete(self, doc: str) -> bool:
        with self.lock:
            found = self.store.delete_document(doc)
        if found and self.uploads_dir and (self.uploads_dir / doc).exists():
            (self.uploads_dir / doc).unlink()
        return found

    def documents(self) -> list[dict]:
        with self.lock:
            return self.store.documents()
