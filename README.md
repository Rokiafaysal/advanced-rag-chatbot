# 🤖 Advanced RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) system that lets you chat with any PDF document using state-of-the-art retrieval and reranking techniques.

---

## 🎯 What it does

Upload any PDF and ask questions about it in natural language. The system retrieves the most relevant information and generates accurate, grounded answers.

---

## ⚡ Key Features

- **Advanced RAG Pipeline** — Hybrid retrieval with FAISS vector search
- **Reranking** — FlagEmbedding reranker for higher precision results
- **RAGAS Evaluation** — Quantitative performance metrics
- **Dynamic PDF Upload** — Works with any PDF, not just one document
- **Clean Gradio UI** — Easy to use interface with evaluation dashboard

---

## 📊 Evaluation Results (RAGAS)

| Metric | Score | Rating |
|--------|-------|--------|
| Faithfulness | 1.00 | 🟢 Excellent |
| Answer Relevancy | 0.85 | 🟢 Good |
| Context Precision | 1.00 | 🟢 Excellent |

> Evaluated on 20 questions from the *Attention Is All You Need* paper

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Reranker | FlagEmbedding (BAAI/bge-reranker-base) |
| Vector Store | FAISS |
| PDF Processing | PyPDF |
| Evaluation | RAGAS |
| UI | Gradio |

---

## 🚀 How it works

1. Upload any PDF document
2. System splits it into chunks and creates embeddings
3. Your question is embedded and matched against chunks using FAISS
4. Top 10 candidates are reranked by FlagEmbedding
5. Best 3 chunks are sent to the LLM with your question
6. LLM generates a grounded answer based only on the PDF content

---

## 📦 Installation

```bash
pip install pypdf faiss-cpu sentence-transformers groq gradio ragas FlagEmbedding
```

---

## 🔑 Setup

```python
# Add your Groq API key
GROQ_API_KEY = "your_key_here"
```

---

## 💡 Use Cases

- 📚 Ask questions about research papers
- 🏥 Extract information from medical reports  
- 📋 Analyze business documents and contracts
- 🎓 Study any book or lecture notes

---

## 👩‍💻 Author

**Rokia Faysal** — LLM & GenAI Engineer
[GitHub](https://github.com/Rokiafaysal) 
