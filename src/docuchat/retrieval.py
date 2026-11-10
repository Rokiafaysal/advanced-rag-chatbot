import faiss
import numpy as np

from docuchat.config import settings
from docuchat.gpu import DEVICE, gpu
from docuchat.models import Chunk, ScoredChunk

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(settings.embed_model, device=DEVICE)
    return _embedder


@gpu
def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(get_embedder().encode(texts, batch_size=32), dtype="float32")


def build_index(chunks: list[Chunk]) -> faiss.Index:
    emb = embed([c.text for c in chunks])
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index


def dense_search(
    query: str, chunks: list[Chunk], index: faiss.Index, k: int | None = None
) -> list[ScoredChunk]:
    k = min(k or settings.retrieve_k, len(chunks))
    distances, ids = index.search(embed([query]), k)
    return [
        ScoredChunk(chunk=chunks[i], score=-float(d))
        for d, i in zip(distances[0], ids[0])
        if i != -1
    ]