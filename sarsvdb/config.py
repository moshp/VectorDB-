import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
MANIFEST = DATA / "manifest.json"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"

# Each web visitor gets a private, empty database for their own uploads,
# deleted after SESSION_TTL_HOURS without activity.
SESSIONS_DIR = Path(os.environ.get("DQA_SESSIONS_DIR", DATA / "sessions"))
SESSION_TTL_HOURS = float(os.environ.get("DQA_SESSION_TTL_HOURS", "6"))
MAX_DOCS_PER_SESSION = int(os.environ.get("DQA_MAX_DOCS", "5"))
MAX_PAGES_PER_DOC = int(os.environ.get("DQA_MAX_PAGES", "150"))
MAX_UPLOAD_MB = int(os.environ.get("DQA_MAX_UPLOAD_MB", "20"))
# Part 3 ingests the SARS guides into its own test database.
EVAL_STORE = DATA / "eval_store"
EVAL_CHROMA = DATA / "eval_chroma"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40
# Chunks stay well inside the embedding model's 512-token limit.
MAX_CHUNK_TOKENS = 254

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 64

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_K = 20
CONTEXT_K = 5

LLM_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLM_PATH = Path(os.environ.get("DQA_MODELS_DIR", ROOT / "models")) / LLM_FILE
LLM_CONTEXT = 4096
LLM_MAX_ANSWER_TOKENS = 400
