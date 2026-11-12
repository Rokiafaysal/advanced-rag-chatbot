
import json
import sys
import time
from pathlib import Path

from docuchat.chunking import chunk_pages
from docuchat.cleaning import clean_pages
from docuchat.config import settings
from docuchat.ingestion import load_pdf
from docuchat.retrieval import build_index, dense_search, get_embedder

KS = (5, 20)
QUESTIONS = Path(__file__).parent / "questions.jsonl"


def first_hit_rank(results, gold_pages: set[int]) -> int | None:
    for rank, r in enumerate(results, 1):
        if r.chunk.page in gold_pages:
            return rank
    return None


def main(docs_dir: str = ".") -> None:
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]

    print(f"model: {settings.embed_model}")
    get_embedder()  # load the model before timing

    docs = {}
    for name in sorted({q["doc"] for q in questions}):
        chunks = chunk_pages(clean_pages(load_pdf(Path(docs_dir) / name)), name)
        start = time.perf_counter()
        index = build_index(chunks)
        print(f"{name}: {len(chunks)} chunks embedded in {time.perf_counter() - start:.1f}s")
        docs[name] = (chunks, index)

    hits = {k: 0 for k in KS}
    print()
    for q in questions:
        chunks, index = docs[q["doc"]]
        results = dense_search(q["question"], chunks, index, k=max(KS))
        rank = first_hit_rank(results, set(q["pages"]))
        for k in KS:
            if rank is not None and rank <= k:
                hits[k] += 1
        print(f"{q['id']:<9} {q['type']:<14} rank {rank if rank else '-':>3}")

    n = len(questions)
    print("\n" + " | ".join(f"Hit@{k}: {hits[k]}/{n} ({hits[k] / n:.0%})" for k in KS))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")