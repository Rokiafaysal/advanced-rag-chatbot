"""Gradio UI for DocuChat. All RAG logic lives in docuchat.pipeline."""
import gradio as gr

from docuchat.config import settings
from docuchat.generation import GenerationError
from docuchat.gpu import ZERO_GPU
from docuchat.ingestion import IngestionError
from docuchat.pipeline import ask, ingest
from docuchat.reranking import get_reranker
from docuchat.retrieval import get_embedder

if ZERO_GPU:
    # Load models at startup so the first question isn't slow.
    get_embedder()
    get_reranker()


# ---------- handlers (state is per browser session) ----------

def process_pdf(pdf_path, _doc):
    if not pdf_path:
        return None, "⬆️ Upload a PDF to get started", [], ""
    try:
        doc = ingest(pdf_path)
    except IngestionError as e:
        raise gr.Error(str(e))
    status = f"✅ **{doc.name}** loaded: {doc.n_pages} pages with text, {len(doc.chunks)} chunks. Ask away!"
    return doc, status, [], ""


def format_sources(sources) -> str:
    return "\n\n".join(
        f"**#{i} · page {r.chunk.page} · rerank score {r.score:.2f}**\n\n"
        f"> {r.chunk.text[:600]}{'…' if len(r.chunk.text) > 600 else ''}"
        for i, r in enumerate(sources, 1)
    )


def chat(question, history, doc):
    question = (question or "").strip()
    if not question:
        return history, "", gr.update()
    if doc is None:
        raise gr.Error("Upload a PDF first.")
    try:
        answer, sources = ask(question, doc, history)
    except GenerationError as e:
        print(f"LLM error: {e.__cause__!r}")
        raise gr.Error(str(e))
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text},
    ]
    return history, "", format_sources(sources)


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
1. The PDF is split into {settings.chunk_size}-word chunks with {settings.chunk_overlap}-word overlap and embedded with **{settings.embed_model}**.
2. Your question is matched against the chunks with **FAISS**, keeping the top {settings.retrieve_k} candidates.
3. A **BGE cross-encoder** ({settings.rerank_model}) reranks them and keeps the best {settings.top_n}.
4. **{settings.groq_model}** (via Groq) answers using only those chunks. Open *Retrieved sources* to see exactly what it read.

[Source code on GitHub](https://github.com/Rokiafaysal/advanced-rag-chatbot)
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Advanced RAG Chatbot") as demo:
    doc_state = gr.State(None)
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
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                with gr.Accordion("📎 Retrieved sources", open=False):
                    sources_md = gr.Markdown()

    with gr.Tab("📊 Evaluation"):
        gr.Markdown(EVAL_MD)
    with gr.Tab("⚙️ How it works"):
        gr.Markdown(HOW_MD)

    pdf.change(process_pdf, [pdf, doc_state], [doc_state, status, chatbot, sources_md])
    ask_btn.click(chat, [question, chatbot, doc_state], [chatbot, question, sources_md])
    question.submit(chat, [question, chatbot, doc_state], [chatbot, question, sources_md])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()