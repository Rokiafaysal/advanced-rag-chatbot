

import os

import faiss
import gradio as gr
import numpy as np

from docuchat.config import settings
try:
    import spaces
    ZERO_GPU = True
except ImportError:
    ZERO_GPU = False
from docuchat.chunking import chunk_pages
from docuchat.ingestion import IngestionError, load_pdf
from docuchat.models import Chunk, ScoredChunk

USE_GPU = settings.use_gpu

DEVICE = "cuda" if USE_GPU else "cpu"

if ZERO_GPU:
    gpu = spaces.GPU(duration=60) if USE_GPU else (lambda fn: fn)

    if not USE_GPU:
        # Required for ZeroGPU Spaces
        @spaces.GPU
        def _gpu_placeholder():
            return True
else:
    gpu = lambda fn: fn


EMBED_MODEL = settings.embed_model
RERANK_MODEL = settings.rerank_model
LLM_MODEL = settings.groq_model

CHUNK_SIZE = settings.chunk_size     
CHUNK_OVERLAP = settings.chunk_overlap  
RETRIEVE_K = settings.retrieve_k     
TOP_N = settings.top_n            
MAX_PAGES = settings.max_pages     

SYSTEM_PROMPT = (
    "You answer questions about a document using only the context provided. "
    "If the answer is not in the context, say you couldn't find it in the document. "
    "Be clear and concise."
)



_models = {}


def get_embedder():
    if "embedder" not in _models:
        from sentence_transformers import SentenceTransformer
        _models["embedder"] = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    return _models["embedder"]


def get_reranker():
    if "reranker" not in _models:
        from sentence_transformers import CrossEncoder
        _models["reranker"] = CrossEncoder(RERANK_MODEL, max_length=512, device=DEVICE)
    return _models["reranker"]

def get_llm():
    from groq import Groq
    if settings.groq_api_key is None:
        raise gr.Error("GROQ_API_KEY is not set. Add it in the Space settings under Secrets.")
    if "llm" not in _models:
        _models["llm"] = Groq(api_key=settings.groq_api_key.get_secret_value())
    return _models["llm"]


if ZERO_GPU:
    get_embedder()
    get_reranker()



@gpu
def embed(texts):
    return np.asarray(get_embedder().encode(texts, batch_size=32), dtype="float32")


@gpu
def rerank_scores(query, candidates):
    return [float(s) for s in get_reranker().predict([[query, c] for c in candidates])]

def build_index(chunks: list[Chunk]):
    emb = embed([c.text for c in chunks])
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index


def retrieve(query: str, chunks: list[Chunk], index) -> list[ScoredChunk]:
    _, ids = index.search(embed([query]), min(RETRIEVE_K, len(chunks)))
    candidates = [chunks[i] for i in ids[0] if i != -1]
    scores = rerank_scores(query, [c.text for c in candidates])
    ranked = sorted(
        (ScoredChunk(chunk=c, score=s) for c, s in zip(candidates, scores)),
        key=lambda r: r.score,
        reverse=True,
    )
    return ranked[:TOP_N]

def generate(query, context_chunks, history):
    context = "\n\n---\n\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-4:]:  # short memory for follow-up questions
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Context from the document:\n{context}\n\nQuestion: {query}"})
    kwargs = dict(model=LLM_MODEL, messages=messages, temperature=0.3, max_completion_tokens=2048)
    if LLM_MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"  # reasoning models: keep answers fast
    try:
        resp = get_llm().chat.completions.create(**kwargs)
    except Exception as e:
        print(f"LLM error: {e!r}")
        raise gr.Error("The language model didn't respond. Please try again in a moment.")
    answer = (resp.choices[0].message.content or "").strip()
    return answer or "I couldn't generate an answer. Please try rephrasing the question."


def process_pdf(pdf_path, _state):
    if not pdf_path:
        return None, "⬆️ Upload a PDF to get started", [], ""
    name = os.path.basename(pdf_path)
    try:
        pages = load_pdf(pdf_path)
    except IngestionError as e:
        raise gr.Error(str(e))
    chunks = chunk_pages(pages, source=name)
    index = build_index(chunks)
    status = f"✅ **{name}** loaded: {len(pages)} pages with text, {len(chunks)} chunks. Ask away!"
    return {"chunks": chunks, "index": index}, status, [], ""

def chat(question, history, state):
    question = (question or "").strip()
    if not question:
        return history, "", gr.update()
    if not state:
        raise gr.Error("Upload a PDF first.")
    ranked = retrieve(question, state["chunks"], state["index"])
    answer = generate(question, [r.chunk.text for r in ranked], history)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    sources = "\n\n".join(
        f"**#{i} · page {r.chunk.page} · rerank score {r.score:.2f}**\n\n"
        f"> {r.chunk.text[:600]}{'…' if len(r.chunk.text) > 600 else ''}"
        for i, r in enumerate(ranked, 1)
    )
    return history, "", sources

# ---------- UI ----------

EVAL_MD = """
## RAGAS evaluation (offline)

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer relevancy | 0.85 |
| Context precision | 1.00 |

Evaluated offline on questions from *Attention Is All You Need*, with Llama 3.3 70B as the generator.
RAGAS runs separately from the live app because LLM-judged evaluation takes minutes per question.
"""

HOW_MD = f"""
## How it works
1. The PDF is split into {CHUNK_SIZE}-word chunks with {CHUNK_OVERLAP}-word overlap and embedded with **all-MiniLM-L6-v2**.
2. Your question is matched against the chunks with **FAISS**, keeping the top {RETRIEVE_K} candidates.
3. A **BGE cross-encoder** ({RERANK_MODEL}) reranks them and keeps the best {TOP_N}.
4. **{LLM_MODEL}** (via Groq) answers using only those chunks. Open *Retrieved sources* to see exactly what it read.

[Source code on GitHub](https://github.com/Rokiafaysal/advanced-rag-chatbot)
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Advanced RAG Chatbot") as demo:
    state = gr.State(None)
    gr.Markdown("# 🤖 Advanced RAG Chatbot\nUpload a PDF and ask questions. Answers are grounded in your document, with sources shown.")

    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                pdf = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
                status = gr.Markdown("⬆️ Upload a PDF to get started")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", height=420, label="Chat")
                with gr.Row():
                    question = gr.Textbox(placeholder="Ask something about your PDF…", show_label=False, scale=5)
                    ask = gr.Button("Ask", variant="primary", scale=1)
                with gr.Accordion("📎 Retrieved sources", open=False):
                    sources = gr.Markdown()

    with gr.Tab("📊 Evaluation"):
        gr.Markdown(EVAL_MD)
    with gr.Tab("⚙️ How it works"):
        gr.Markdown(HOW_MD)

    pdf.change(process_pdf, [pdf, state], [state, status, chatbot, sources])
    ask.click(chat, [question, chatbot, state], [chatbot, question, sources])
    question.submit(chat, [question, chatbot, state], [chatbot, question, sources])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
