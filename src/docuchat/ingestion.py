from pathlib import Path
from pypdf import PdfReader

from  docuchat.config import settings


class IngestionError(Exception):
    """The file can't be used: not a PDF, too many pages, or no text."""


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Page numbers start at 1."""
    try:
        reader = PdfReader(path)
        n_pages = len(reader.pages)  # pypdf only parses the file here
    except Exception as e:
        raise IngestionError("Couldn't read this file. Please upload a valid PDF.") from e

    if n_pages > settings.max_pages:
        if len(reader.pages) > settings.max_pages:
            raise IngestionError(f"This demo supports PDFs up to {settings.max_pages} pages.")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))

    if not pages:
        raise IngestionError("No text found. Scanned PDFs (images only) aren't supported.")
    return pages