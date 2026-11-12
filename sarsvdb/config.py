from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
MANIFEST = DATA / "manifest.json"
PAGES_FILE = DATA / "pages.jsonl"
CHUNKS_FILE = DATA / "chunks.jsonl"
STORE_DIR = DATA / "store"
CHROMA_DIR = DATA / "chroma"
REPORTS = ROOT / "reports"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40
# The model reads 256 tokens including [CLS] and [SEP].
MAX_CHUNK_TOKENS = 254

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 64
