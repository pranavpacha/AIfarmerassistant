"""End-to-end pipeline: translate query -> retrieve -> generate -> translate
answer back. This is the single function both app.py and eval/evaluate.py call,
so the UI and the evaluation harness are always testing the exact same logic."""
from dataclasses import dataclass, field

from model.translate import to_english, from_english
from model.agriparam_client import get_agriparam
from rag.retriever import get_retriever

TOP_K = 4


@dataclass
class AdvisoryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    english_query: str = ""
    english_answer: str = ""


def ask(question: str, language: str = "English", top_k: int = TOP_K) -> AdvisoryResult:
    english_query = to_english(question, language)

    retriever = get_retriever()
    sources = retriever.search(english_query, top_k=top_k)

    context = "\n".join(
        f"- [{s['crop']} / {s['category']}] Q: {s['query']} A: {s['answer']}"
        for s in sources
    )

    agriparam = get_agriparam()
    english_answer = agriparam.answer(english_query, context)

    final_answer = from_english(english_answer, language)

    return AdvisoryResult(
        answer=final_answer,
        sources=sources,
        english_query=english_query,
        english_answer=english_answer,
    )
