"""Token-based chunking that respects lines and section headings."""
import re
from dataclasses import dataclass
from itertools import groupby

LIST_ITEM = re.compile(r"^\(?([A-Za-z]{1,4}|\d{1,3})[).]\s|^[•\-–▪]\s?")
HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+[A-Z][^.]{2,80}|[A-Z][A-Z0-9 ,&()/\-]{4,80})$")


@dataclass
class Unit:
    text: str
    page: int
    n_tokens: int
    heading: bool


def _units(pages: list[dict], tokenizer, chunk_tokens: int) -> list[Unit]:
    units = []
    for p in pages:
        for line in p["text"].splitlines():
            line = line.strip()
            if not line:
                continue
            enc = tokenizer(line, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            if len(offsets) <= chunk_tokens:
                units.append(Unit(line, p["page"], len(offsets), bool(HEADING.match(line))))
                continue
            for start in range(0, len(offsets), chunk_tokens):
                window = offsets[start:start + chunk_tokens]
                piece = line[window[0][0]:window[-1][1]]
                units.append(Unit(piece, p["page"], len(window), False))
    return units


def _overlap(units: list[Unit], overlap_tokens: int) -> list[Unit]:
    """The last lines of a chunk, about overlap_tokens long, to repeat at the
    start of the next one. If those lines are items of a list, the list's
    intro line ('...as follows:') comes with them, up to twice the budget,
    so items such as 'A) Due on 30 Nov 2025' never lose what they refer to."""
    tail, t = [], 0
    for u in reversed(units):
        if t + u.n_tokens > overlap_tokens:
            break
        tail.insert(0, u)
        t += u.n_tokens
    if not tail or not LIST_ITEM.match(tail[0].text):
        return tail
    start = len(units) - len(tail)
    i = start - 1
    while i >= 0 and LIST_ITEM.match(units[i].text) and not units[i].text.rstrip().endswith(":"):
        i -= 1
    if i >= 0 and units[i].text.rstrip().endswith(":"):
        extended = units[i:]
        if sum(u.n_tokens for u in extended) <= 2 * overlap_tokens:
            return extended
    return tail


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chunk_document(pages: list[dict], tokenizer, chunk_tokens: int,
                   overlap_tokens: int, max_tokens: int) -> list[dict]:
    """Packs lines into chunks of about chunk_tokens, carrying the last
    overlap_tokens worth of lines into the next chunk. A heading starts a new
    chunk (without overlap) once the current one is at least half full."""
    meta = {k: pages[0][k] for k in ("doc", "title", "category", "url")}
    chunks: list[dict] = []
    current: list[Unit] = []
    has_new = False

    def emit(units: list[Unit]) -> None:
        units = list(units)
        text = "\n".join(u.text for u in units)
        n = _token_count(tokenizer, text)
        while n > max_tokens and len(units) > 1:
            units.pop(0)
            text = "\n".join(u.text for u in units)
            n = _token_count(tokenizer, text)
        chunks.append({
            **meta,
            "chunk_id": f"{meta['doc']}::{len(chunks)}",
            "page_start": units[0].page,
            "page_end": units[-1].page,
            "n_tokens": n,
            "text": text,
        })

    for unit in _units(pages, tokenizer, chunk_tokens):
        size = sum(u.n_tokens for u in current)
        if current and has_new and unit.heading and size >= chunk_tokens // 2:
            emit(current)
            current, has_new = [], False
        elif current and has_new and size + unit.n_tokens > chunk_tokens:
            emit(current)
            current, has_new = _overlap(current, overlap_tokens), False
        current.append(unit)
        has_new = True

    if current and has_new:
        emit(current)
    return chunks


def chunk_pages(pages: list[dict], tokenizer, chunk_tokens: int,
                overlap_tokens: int, max_tokens: int) -> list[dict]:
    chunks = []
    for _, doc_pages in groupby(pages, key=lambda p: p["doc"]):
        chunks.extend(chunk_document(list(doc_pages), tokenizer,
                                     chunk_tokens, overlap_tokens, max_tokens))
    return chunks
