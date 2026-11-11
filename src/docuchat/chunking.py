
import re
from functools import lru_cache

from tokenizers import Tokenizer

from docuchat.config import settings
from docuchat.models import Chunk

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(settings.rerank_model)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def _split_long(sentence: str, max_tokens: int) -> list[str]:
    """Cut a sentence that's over the budget by words (rare: long tables or lists)."""
    pieces, current = [], []
    for word in sentence.split():
        if current and count_tokens(" ".join(current + [word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _sentences(text: str, max_tokens: int) -> list[str]:
    units = []
    for paragraph in PARAGRAPH_BREAK.split(text):
        for sentence in SENTENCE_END.split(paragraph.strip()):
            if not sentence:
                continue
            if count_tokens(sentence) <= max_tokens:
                units.append(sentence)
            else:
                units.extend(_split_long(sentence, max_tokens))
    return units


def _pack(units: list[str], max_tokens: int, overlap: int) -> list[str]:
    chunks, current, size = [], [], 0
    for unit in units:
        n = count_tokens(unit)
        if current and size + n > max_tokens:
            chunks.append(" ".join(current))
            # carry the last whole sentences (up to `overlap` tokens) into the next chunk
            carried, carried_size = [], 0
            for prev in reversed(current):
                t = count_tokens(prev)
                if carried_size + t > overlap:
                    break
                carried.insert(0, prev)
                carried_size += t
            current, size = carried, carried_size
            if size + n > max_tokens:  # no room for the overlap with this sentence
                current, size = [], 0
        current.append(unit)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap

    chunks = []
    for page_number, text in pages:
        for piece in _pack(_sentences(text, size), size, overlap):
            chunks.append(Chunk(id=len(chunks), text=piece, source=source, page=page_number))
    return chunks