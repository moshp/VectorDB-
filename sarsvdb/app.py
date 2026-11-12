"""Web app: upload documents (Part 1) and ask questions about them (Part 2)."""
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .embed import Embedder
from .ingest import IMAGE_SUFFIXES
from .pipeline import Ingestor, safe_name
from .rag import RAG, Generator, Retriever

ALLOWED_SUFFIXES = {".pdf"} | IMAGE_SUFFIXES
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
WEB_DIR = Path(__file__).parent / "web"

state: dict = {}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
# One ingestion at a time: OCR and embedding already use every CPU core.
worker = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder(config.MODEL_NAME, config.QUERY_PREFIX)
    ingestor = Ingestor(config.APP_STORE, embedder, uploads_dir=config.UPLOADS)
    state["ingestor"] = ingestor
    state["rag"] = RAG(ingestor, Retriever(ingestor), Generator())
    yield
    worker.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Document Q&A", lifespan=lifespan)


def _update(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def _run_ingest(job_id: str, path: Path, filename: str) -> None:
    _update(job_id, status="running")
    try:
        result = state["ingestor"].ingest(
            path, filename, progress=lambda stage, frac: _update(job_id, stage=stage, progress=frac))
        _update(job_id, status="done", progress=1.0, result=result)
    except Exception as e:  # reported to the user on the upload card
        _update(job_id, status="error", error=str(e))
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return state["ingestor"].documents()


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)) -> list[dict]:
    created = []
    for f in files:
        name = safe_name(f.filename or "document")
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            created.append({"filename": name, "status": "error",
                            "error": "Only PDF and image files (PNG, JPG, TIFF, BMP, WebP) can be uploaded."})
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
            created.append({"filename": name, "status": "error",
                            "error": f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."})
            continue
        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "filename": name, "status": "queued",
                            "stage": "Waiting to start", "progress": 0.0}
        worker.submit(_run_ingest, job_id, path, name)
        created.append(jobs[job_id])
    return created


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Unknown upload")
        return dict(jobs[job_id])


@app.delete("/api/documents/{doc}")
def delete_document(doc: str) -> dict:
    if not state["ingestor"].delete(doc):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc}


@app.get("/files/{doc}")
def original_file(doc: str) -> FileResponse:
    names = {d["doc"] for d in state["ingestor"].documents()}
    path = config.UPLOADS / doc
    if doc not in names or not path.is_file():
        raise HTTPException(404, "Document not found")
    return FileResponse(path, filename=doc, content_disposition_type="inline")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    docs: list[str] | None = None


@app.post("/api/ask")
def ask(q: Question) -> dict:
    return state["rag"].answer(q.question.strip(), docs=q.docs or None)
