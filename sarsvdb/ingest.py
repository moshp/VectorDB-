"""PDF text extraction, page by page, with repeated headers and footers removed."""
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

EDGE_LINES = 3


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def clean_pages(pages: list[str]) -> list[str]:
    """Drops lines that recur on at least half the pages (running headers,
    footers, page numbers; pypdf sometimes places these mid-page) and
    rejoins hyphenated words."""
    lines_per_page = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
    boilerplate: set[str] = set()
    if len(pages) >= 4:
        counts = Counter()
        for lines in lines_per_page:
            counts.update({_norm(l) for l in lines})
        threshold = max(3, len(pages) // 2)
        boilerplate = {l for l, c in counts.items() if c >= threshold}

    cleaned = []
    for lines in lines_per_page:
        kept = [l for l in lines if _norm(l) not in boilerplate]
        text = "\n".join(re.sub(r"[ \t ]+", " ", l) for l in kept)
        text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
        cleaned.append(text)
    return cleaned


def extract_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    raw = []
    for page in reader.pages:
        try:
            raw.append(page.extract_text() or "")
        except Exception:
            raw.append("")
    return clean_pages(raw)


def ingest(manifest_path: Path, pdf_dir: Path, out_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    n_pages = 0
    with out_path.open("w", encoding="utf-8") as out:
        for name, meta in sorted(manifest.items()):
            if not meta.get("usable", True):
                continue
            pages = extract_pdf(pdf_dir / name)
            kept = 0
            for i, text in enumerate(pages, start=1):
                if len(text.strip()) < 20:
                    continue
                out.write(json.dumps({
                    "doc": name,
                    "title": meta.get("title") or name,
                    "category": meta.get("category"),
                    "url": meta.get("url"),
                    "page": i,
                    "text": text,
                }, ensure_ascii=False) + "\n")
                kept += 1
            n_pages += kept
            print(f"  {name}: {kept}/{len(pages)} pages with text")
    return n_pages
