from docuchat.config import settings
from docuchat.gpu import DEVICE, gpu
from docuchat.models import Chunk, ScoredChunk


_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(settings.rerank_model, max_length=512, device=DEVICE)
    return _reranker

@gpu
def _scores( query:str , texts :str)->list[float]:
    return [float(s)  for s in get_reranker().predict([[query, t] for t in texts])]



def rerank(
    query: str, candidates: list[ScoredChunk], top_n: int | None = None
) -> list[ScoredChunk]:
    if not candidates:
        return []
    top_n = top_n or settings.top_n
    scores = _scores(query, [c.chunk.text for c in candidates])
    ranked = sorted(
        (ScoredChunk(chunk=c.chunk, score=s) for c, s in zip(candidates, scores)),
        key=lambda r: r.score,
        reverse=True,
    )
    return ranked[:top_n]