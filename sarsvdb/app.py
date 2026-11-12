"""Web app: upload documents (Part 1) and ask questions about them (Part 2).

Each visitor gets a private session (a cookie) with its own empty vector
database and uploads folder. Sessions are deleted after SESSION_TTL_HOURS
without activity. The embedding model, re-ranker and Llama are shared.
"""
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from . import config
from .embed import Embedder
from .ingest import IMAGE_SUFFIXES
from .pipeline import Ingestor, safe_name
from .rag import RAG, Generator, Retriever, load_reranker

ALLOWED_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES
MAX_UPLOAD_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
WEB_DIR = Path(__file__).parent / "web"
COOKIE = "dqa_session"
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
SWEEP_SECONDS = 600

models: dict = {}
sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
# One ingestion at a time: OCR and embedding already use every CPU core.
worker = ThreadPoolExecutor(max_workers=1)


class Session:
    def __init__(self, sid: str):
        self.sid = sid
        root = config.SESSIONS_DIR / sid
        self.ingestor = Ingestor(root / "store", models["embedder"], uploads_dir=root / "uploads")
        self.rag = RAG(self.ingestor, Retriever(self.ingestor, models["reranker"]), models["generator"])
        self.last_seen = time.time()


def _sweep_forever(stop: threading.Event) -> None:
    while not stop.wait(SWEEP_SECONDS):
        cutoff = time.time() - config.SESSION_TTL_HOURS * 3600
        with sessions_lock:
            expired = [sid for sid, s in sessions.items() if s.last_seen < cutoff]
            for sid in expired:
                del sessions[sid]
        with jobs_lock:
            for job_id in [j for j, job in jobs.items() if job["sid"] in expired]:
                del jobs[job_id]
        for sid in expired:
            shutil.rmtree(config.SESSIONS_DIR / sid, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sessions live only as long as the server process.
    shutil.rmtree(config.SESSIONS_DIR, ignore_errors=True)
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    models["embedder"] = Embedder(config.MODEL_NAME, config.QUERY_PREFIX)
    models["reranker"] = load_reranker()
    models["generator"] = Generator()
    stop = threading.Event()
    threading.Thread(target=_sweep_forever, args=(stop,), daemon=True).start()
    yield
    stop.set()
    worker.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Document Q&A", lifespan=lifespan)


def session(request: Request, response: Response) -> Session:
    sid = request.cookies.get(COOKIE, "")
    with sessions_lock:
        if not SESSION_ID.match(sid) or sid not in sessions:
            sid = uuid.uuid4().hex
            sessions[sid] = Session(sid)
        s = sessions[sid]
        s.last_seen = time.time()
    response.set_cookie(COOKIE, sid, max_age=int(config.SESSION_TTL_HOURS * 3600),
                        httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return s


def _update(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "sid"}


def _run_ingest(job_id: str, s: Session, path: Path, filename: str) -> None:
    _update(job_id, status="running")
    try:
        result = s.ingestor.ingest(
            path, filename, progress=lambda stage, frac: _update(job_id, stage=stage, progress=frac))
        _update(job_id, status="done", progress=1.0, result=result)
    except Exception as e:  # reported to the visitor on the upload card
        _update(job_id, status="error", error=str(e))
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


def _too_many_pages(path: Path) -> bool:
    if path.suffix.lower() != ".pdf":
        return False
    try:
        return len(PdfReader(str(path)).pages) > config.MAX_PAGES_PER_DOC
    except Exception:
        return False  # unreadable PDFs are reported by ingestion itself


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/limits")
def limits() -> dict:
    return {"max_docs": config.MAX_DOCS_PER_SESSION, "max_pages": config.MAX_PAGES_PER_DOC,
            "max_upload_mb": config.MAX_UPLOAD_MB, "ttl_hours": config.SESSION_TTL_HOURS}


@app.get("/api/documents")
def list_documents(s: Session = Depends(session)) -> list[dict]:
    return s.ingestor.documents()


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...), s: Session = Depends(session)) -> list[dict]:
    created = []
    with jobs_lock:
        pending = sum(1 for j in jobs.values() if j["sid"] == s.sid and j["status"] in ("queued", "running"))
    slots = config.MAX_DOCS_PER_SESSION - len(s.ingestor.documents()) - pending
    for f in files:
        name = safe_name(f.filename or "document")
        suffix = Path(name).suffix.lower()

        def reject(message: str) -> None:
            created.append({"filename": name, "status": "error", "error": message})

        if suffix not in ALLOWED_SUFFIXES:
            reject("Only PDF and image files (PNG, JPG, TIFF, BMP, WebP) can be uploaded.")
            continue
        if slots <= 0:
            reject(f"You can have up to {config.MAX_DOCS_PER_SESSION} documents. Delete one to add another.")
            continue
        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / name
        size = 0
        with path.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    break
                out.write(chunk)
        if size > MAX_UPLOAD_BYTES:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            reject(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
            continue
        if _too_many_pages(path):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            reject(f"PDF has more than {config.MAX_PAGES_PER_DOC} pages.")
            continue
        slots -= 1
        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "sid": s.sid, "filename": name, "status": "queued",
                            "stage": "Waiting to start", "progress": 0.0}
            created.append(_public(jobs[job_id]))
        worker.submit(_run_ingest, job_id, s, path, name)
    return created


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, s: Session = Depends(session)) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job["sid"] != s.sid:
            raise HTTPException(404, "Unknown upload")
        return _public(job)


@app.delete("/api/documents/{doc}")
def delete_document(doc: str, s: Session = Depends(session)) -> dict:
    if not s.ingestor.delete(doc):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc}


@app.get("/files/{doc}")
def original_file(doc: str, s: Session = Depends(session)) -> FileResponse:
    names = {d["doc"] for d in s.ingestor.documents()}
    path = s.ingestor.uploads_dir / doc
    if doc not in names or not path.is_file():
        raise HTTPException(404, "Document not found")
    return FileResponse(path, filename=doc, content_disposition_type="inline")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.post("/api/ask")
def ask(q: Question, s: Session = Depends(session)) -> dict:
    return s.rag.answer(q.question.strip())
