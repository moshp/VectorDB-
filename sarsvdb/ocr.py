"""OCR with Tesseract, for scanned PDF pages and images."""
import subprocess
import tempfile
from pathlib import Path

import pytesseract
from PIL import Image

OCR_DPI = 300


def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image.convert("L"), lang="eng")


def render_pdf_page(pdf: Path, page_number: int, dpi: int = OCR_DPI) -> Image.Image:
    """Renders one 1-indexed PDF page to an image with poppler's pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-f", str(page_number), "-l", str(page_number), "-r", str(dpi),
             "-png", "-singlefile", str(pdf), str(prefix)],
            check=True, capture_output=True, timeout=120,
        )
        with Image.open(f"{prefix}.png") as img:
            return img.copy()


def ocr_pdf_page(pdf: Path, page_number: int) -> str:
    return ocr_image(render_pdf_page(pdf, page_number))
