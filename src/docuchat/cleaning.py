"""Clean extracted page text: whitespace, repeated headers/footers, page numbers."""
import re
from collections import Counter
import unicodedata
from docuchat.normalization import TATWEEL

EDGE_LINES = 3   # headers/footers live in the first/last few lines of a page
MIN_REPEATS = 3  # a line on this many page edges is a header/footer, not content

SPACES = re.compile(r"[ \t\u00a0]+")
DIGITS = re.compile(r"\d+")
PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
BLANK_RUNS = re.compile(r"\n{3,}")




def _split_lines(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
    return [SPACES.sub(" ", line).strip() for line in text.splitlines()]

def _key(line: str) -> str:
    # "Page 3 of 8" and "Page 4 of 8" count as the same line
    return DIGITS.sub("#", line)


def _edge_indexes(lines: list[str]) -> set[int]:
    filled = [i for i, line in enumerate(lines) if line]
    return set(filled[:EDGE_LINES] + filled[-EDGE_LINES:])


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    split = [(number, _split_lines(text)) for number, text in pages]

    counts = Counter()
    for _, lines in split:
        counts.update({_key(lines[i]) for i in _edge_indexes(lines)})
    repeated = {key for key, c in counts.items() if c >= MIN_REPEATS}

    cleaned = []
    for number, lines in split:
        edges = _edge_indexes(lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (i in edges and (_key(line) in repeated or PAGE_NUMBER.match(line)))
        ]
        text = BLANK_RUNS.sub("\n\n", "\n".join(kept)).strip()
        if text:
            cleaned.append((number, text))
    return cleaned