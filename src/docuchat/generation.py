from groq import APIError, Groq

from docuchat.config import settings
from docuchat.models import ScoredChunk

SYSTEM_PROMPT = (
    "You answer questions about a document using only the context provided. "
    "If the answer is not in the context, say you couldn't find it in the document. "
    "Be clear and concise."
)


class GenerationError(Exception):
    """The LLM service couldn't produce an answer."""


_client = None


def get_client() -> Groq:
    global _client
    if settings.groq_api_key is None:
        raise GenerationError("GROQ_API_KEY is not set. Add it to .env, Kaggle Secrets or the Space secrets.")
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key.get_secret_value())
    return _client


def build_messages(question: str, chunks: list[ScoredChunk], history: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(r.chunk.text for r in chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history[-4:]]
    messages.append(
        {"role": "user", "content": f"Context from the document:\n{context}\n\nQuestion: {question}"}
    )
    return messages


def generate(
    question: str,
    chunks: list[ScoredChunk],
    history: list[dict] | None = None,
    temperature: float = 0.3,
) -> str:
    kwargs = dict(
        model=settings.groq_model,
        messages=build_messages(question, chunks, history or []),
        temperature=temperature,
        max_completion_tokens=2048,
    )
    if settings.groq_model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"  # reasoning models: keep answers fast
    try:
        resp = get_client().chat.completions.create(**kwargs)
    except APIError as e:
        raise GenerationError("The language model didn't respond. Please try again in a moment.") from e
    answer = (resp.choices[0].message.content or "").strip()
    return answer or "I couldn't generate an answer. Please try rephrasing the question."