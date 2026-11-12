"""Download SARS guide PDFs from sars.gov.za tax-type pages.

Keeps English guides and FAQs, skips forms, translations, technical specs and
letters, and picks round-robin across pages so each tax type is represented.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from pypdf import PdfReader

BASE = "https://www.sars.gov.za/"
PAGES = [
    "types-of-tax/value-added-tax/",
    "types-of-tax/personal-income-tax/",
    "types-of-tax/pay-as-you-earn/",
    "types-of-tax/corporate-income-tax/",
    "types-of-tax/capital-gains-tax/",
    "types-of-tax/donations-tax/",
    "types-of-tax/estate-duty/",
    "types-of-tax/transfer-duty/",
    "types-of-tax/dividends-tax/",
    "types-of-tax/turnover-tax/",
    "customs-and-excise/",
]
KEEP = re.compile(r"guide|faq|interpretation|leaflet", re.I)
SKIP = re.compile(
    r"form|brs|business-requirement|sepedi|sesotho|xitsonga|afr\d|letter|"
    r"presentation|checklist|sticky|look-?feel|industrycommunication|example-of",
    re.I,
)
USER_AGENT = "sars-vectordb-research/1.0"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def pdf_links(page: str) -> list[str]:
    html = fetch(BASE + page).decode("utf-8", errors="replace")
    links = []
    for href in re.findall(r'href="([^"]+\.pdf)"', html, re.I):
        url = urllib.parse.urljoin(BASE + page, href)
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if KEEP.search(name) and not SKIP.search(name) and url not in links:
            links.append(url)
    return links


def text_fingerprint(reader: PdfReader, n_pages: int = 5) -> str:
    """SARS sometimes publishes the same document under two file names, with
    different bytes but identical text; this catches those."""
    text = " ".join((p.extract_text() or "") for p in reader.pages[:n_pages])
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pdfs")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--target-total", type=int, default=55,
                    help="stop once this many usable PDFs exist in --out")
    ap.add_argument("--max-pages", type=int, default=300,
                    help="skip PDFs longer than this so one guide can't dominate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for f in sorted(out.glob("*.pdf")):
        if f.name not in manifest:
            try:
                reader = PdfReader(str(f))
                sample = "".join((p.extract_text() or "") for p in reader.pages[:3])
                title = reader.metadata.title if reader.metadata and reader.metadata.title else f.stem
                n_pages = len(reader.pages)
            except Exception:
                sample, title, n_pages = "", f.stem, 0
            manifest[f.name] = {
                "url": None,
                "category": "earlier scrape",
                "title": title,
                "pages": n_pages,
                "usable": len(sample.strip()) >= 200,
            }

    hashes = {hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob("*.pdf")}
    fingerprints = set()
    for f in out.glob("*.pdf"):
        if manifest.get(f.name, {}).get("usable", True):
            fingerprints.add(text_fingerprint(PdfReader(str(f))))
    seen_urls = {m["url"] for m in manifest.values() if m.get("url")}

    queues = {}
    for page in PAGES:
        try:
            queues[page] = [u for u in pdf_links(page) if u not in seen_urls]
        except Exception as e:
            print(f"SKIP page {page}: {e}")
            queues[page] = []
        print(f"{page}: {len(queues[page])} candidate guides")
        time.sleep(1)

    def usable_count() -> int:
        return sum(1 for m in manifest.values() if m.get("usable", True))

    while usable_count() < args.target_total and any(queues.values()):
        for page in PAGES:
            if usable_count() >= args.target_total:
                break
            if not queues[page]:
                continue
            url = queues[page].pop(0)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            try:
                data = fetch(url)
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                continue
            digest = hashlib.sha256(data).hexdigest()
            if not data.startswith(b"%PDF") or digest in hashes:
                print(f"  SKIP {name}: not a PDF or duplicate content")
                continue
            path = out / re.sub(r"[^A-Za-z0-9._-]+", "-", name)
            path.write_bytes(data)
            try:
                reader = PdfReader(str(path))
                n_pages = len(reader.pages)
                sample = "".join((p.extract_text() or "") for p in reader.pages[:3])
            except Exception as e:
                print(f"  BAD  {name}: {e}")
                path.unlink()
                continue
            if n_pages > args.max_pages or len(sample.strip()) < 200:
                why = "too long" if n_pages > args.max_pages else "no extractable text"
                print(f"  DROP {name}: {n_pages} pages, {why}")
                path.unlink()
                continue
            fingerprint = text_fingerprint(reader)
            if fingerprint in fingerprints:
                print(f"  SKIP {name}: same text as a guide already downloaded")
                path.unlink()
                continue
            fingerprints.add(fingerprint)
            hashes.add(digest)
            manifest[path.name] = {
                "url": url,
                "category": page.strip("/").split("/")[-1],
                "title": Path(name).stem.replace("-", " "),
                "pages": n_pages,
            }
            print(f"  OK   {path.name} ({n_pages} pages) [{usable_count()} usable]")
            manifest_path.write_text(json.dumps(manifest, indent=2))
            time.sleep(1)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Done: {usable_count()} usable PDFs, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
