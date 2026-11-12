"""Text extraction from PDFs and images (OCR where a page has no text layer),
page by page, with repeated headers, footers and contents pages removed."""
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from pypdf import PdfReader

from .ocr import ocr_image, ocr_pdf_page

DOT_LEADER = re.compile(r"(\.\s?){5,}|…{2,}")
TOC_ENTRY = re.compile(r"^\d+(\.\d+)*\.?\s+\S.*\s\d{1,3}$")
TOC_HEADING = re.compile(r"^(table of )?contents$", re.I)


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def drop_contents(lines: list[str]) -> list[str]:
    """Removes table-of-contents lines, which describe a document's layout
    rather than its meaning. Dot-leader lines go everywhere; numbered
    'heading  page' lines only on pages made up mostly of them."""
    if not lines:
        return lines
    toc_like = [bool(DOT_LEADER.search(l) or TOC_ENTRY.match(l)) for l in lines]
    contents_page = sum(toc_like) / len(lines) >= 0.4
    return [
        l for l, t in zip(lines, toc_like)
        if not DOT_LEADER.search(l) and not TOC_HEADING.match(l.strip()) and not (contents_page and t)
    ]


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
        kept = drop_contents([l for l in lines if _norm(l) not in boilerplate])
        text = "\n".join(re.sub(r"[ \t ]+", " ", l) for l in kept)
        text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
        cleaned.append(text)
    return cleaned


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# A page whose text layer has fewer letters and digits than this is treated as scanned.
MIN_TEXT_CHARS = 40


@dataclass
class Extracted:
    title: str
    pages: list[str]          # cleaned text, one entry per page
    ocr_pages: list[int]      # 1-indexed pages that went through OCR


def _letters(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _has_image(page) -> bool:
    """A page with neither text nor images is blank; OCR would find nothing."""
    try:
        return len(page.images) > 0
    except Exception:
        return True


def extract_pdf(path: Path, on_page: Optional[Callable[[int, int, str], None]] = None) -> Extracted:
    """Uses each page's own text layer; pages without one are OCR'd."""
    reader = PdfReader(str(path))
    raw, ocr_pages = [], []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        method = "text"
        if _letters(text) < MIN_TEXT_CHARS and _has_image(page):
            text = ocr_pdf_page(path, i)
            ocr_pages.append(i)
            method = "ocr"
        raw.append(text)
        if on_page:
            on_page(i, total, method)
    meta_title = reader.metadata.title if reader.metadata and reader.metadata.title else ""
    return Extracted(_title(meta_title, path), clean_pages(raw), ocr_pages)


GENERIC_TITLE = re.compile(r"^(table of )?contents|^(part|chapter|section)\s*\w{0,4}$|^untitled|^document\s*\d*$", re.I)


def _title(meta_title: str, path: Path) -> str:
    """The PDF's own title when it names the document; otherwise the file name.
    Metadata titles are often leftovers such as 'CONTENTS' or 'Microsoft Word - x.docx'."""
    title = re.sub(r"^microsoft (word|powerpoint)\s*-\s*", "", meta_title.strip(), flags=re.I)
    title = re.sub(r"\.(docx?|pptx?|pdf)$", "", title, flags=re.I).strip()
    if _letters(title) < 4 or GENERIC_TITLE.match(title):
        return path.stem.replace("_", " ")
    return title


def extract_image(path: Path) -> Extracted:
    with Image.open(path) as img:
        text = ocr_image(img)
    return Extracted(path.stem, clean_pages([text]), [1])


def extract_document(path: Path, on_page: Optional[Callable[[int, int, str], None]] = None) -> Extracted:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, on_page)
    if suffix in IMAGE_SUFFIXES:
        extracted = extract_image(path)
        if on_page:
            on_page(1, 1, "ocr")
        return extracted
    raise ValueError(f"unsupported file type {suffix!r}; upload a PDF or an image")


def fingerprint(pages: list[str]) -> str:
    """Identifies a document by its text, so the same document saved under
    another name, or with different bytes, is recognised as a duplicate."""
    text = re.sub(r"\s+", " ", " ".join(pages)).strip().lower()
    return hashlib.sha256(text.encode()).hexdigest()
