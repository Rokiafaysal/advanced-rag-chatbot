import re
from pathlib import Path

import pymupdf
from pypdf import PdfReader

from docuchat.config import settings

ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTER = re.compile(r"[A-Za-z]")


class IngestionError(Exception):
    """The file can't be used: not a PDF, too many pages, or no text."""


def is_arabic(text: str) -> bool:
    return len(ARABIC_LETTER.findall(text)) > len(LATIN_LETTER.findall(text))


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    try:
        mupdf_doc = pymupdf.open(path, filetype="pdf")
        pypdf_doc = PdfReader(path)
        n_pages = len(pypdf_doc.pages)  # pypdf only parses the file here
    except Exception as e:
        raise IngestionError("Couldn't read this file. Please upload a valid PDF.") from e

    if n_pages > settings.max_pages:
        raise IngestionError(f"This demo supports PDFs up to {settings.max_pages} pages.")

    pages = []
    for i in range(n_pages):
        text = mupdf_doc[i].get_text("text", sort=True)
        if is_arabic(text):
            text = pypdf_doc.pages[i].extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))

    if not pages:
        raise IngestionError("No text found. Scanned PDFs (images only) aren't supported.")
    return pages