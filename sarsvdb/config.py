from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
MANIFEST = DATA / "manifest.json"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"

# The web app's database starts empty and holds only uploaded documents.
APP_STORE = DATA / "app_store"
UPLOADS = DATA / "uploads"
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
LLM_PATH = ROOT / "models" / LLM_FILE
LLM_CONTEXT = 4096
LLM_MAX_ANSWER_TOKENS = 400
