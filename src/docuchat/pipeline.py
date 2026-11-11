from dataclasses import dataclass
from pathlib import Path

import faiss

from docuchat.chunking import chunk_pages
from docuchat.generation import generate
from docuchat.ingestion import load_pdf
from docuchat.models import Answer, Chunk, Citation, ScoredChunk
from docuchat.reranking import rerank
from docuchat.retrieval import build_index, dense_search
from docuchat.cleaning import clean_pages


@dataclass
class Document:
    name: str
    n_pages: int
    chunks: list[Chunk]
    index: faiss.Index


def ingest(path: str | Path) -> Document:
    name = Path(path).name
    pages = clean_pages(load_pdf(path))
    chunks = chunk_pages(pages, source=name)
    return Document(name=name, n_pages=len(pages), chunks=chunks, index=build_index(chunks))


def retrieve(question: str, doc: Document) -> list[ScoredChunk]:
    return rerank(question, dense_search(question, doc.chunks, doc.index))


def ask(
    question: str, doc: Document, history: list[dict] | None = None
) -> tuple[Answer, list[ScoredChunk]]:
    sources = retrieve(question, doc)
    text = generate(question, sources, history)

    citations: list[Citation] = []
    for r in sources:
        c = Citation(source=r.chunk.source, page=r.chunk.page)
        if c not in citations:
            citations.append(c)

    return Answer(text=text, citations=citations), sources