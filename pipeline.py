"""End-to-end pipeline: translate query -> retrieve -> generate -> translate
answer back. This is the single function both app.py and eval/evaluate.py call,
so the UI and the evaluation harness are always testing the exact same logic."""
from dataclasses import dataclass, field

from model.translate import to_english, from_english
from model.agriparam_client import get_agriparam
from rag.retriever import get_retriever

TOP_K = 4

# Below this top-1 retrieval similarity, there is no advisory entry close
# enough to ground an answer in. Matches the threshold eval/evaluate.py uses
# to score the out-of-domain trap cases (t36, t37), which exposed the real
# issue this guards against: without this check, AgriParam generates a
# confident-sounding answer regardless of retrieval quality — for t36/t37 it
# fabricated a stock price and a description of an "income tax" system that
# don't exist anywhere in the corpus. See eval/failure_analysis.md.
LOW_CONFIDENCE_THRESHOLD = 0.35

DECLINE_MESSAGE = (
    "I don't have grounded information about this in my agricultural advisory "
    "database. I can help with crop management, irrigation, pests, fertilizers, "
    "government schemes, and related farming topics — try rephrasing your "
    "question around one of these."
)


@dataclass
class AdvisoryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    english_query: str = ""
    english_answer: str = ""
    grounded: bool = True
    top_score: float = 0.0


def ask(question: str, language: str = "English", top_k: int = TOP_K) -> AdvisoryResult:
    english_query = to_english(question, language)

    retriever = get_retriever()
    sources = retriever.search(english_query, top_k=top_k)
    top_score = sources[0]["score"] if sources else 0.0

    if top_score < LOW_CONFIDENCE_THRESHOLD:
        # No generation call at all: nothing in the corpus is close enough to
        # ground an answer, so there is nothing useful for AgriParam to say.
        # Skipping the call also avoids spending a ~minute of GPU time on a
        # query we already know we can't ground.
        return AdvisoryResult(
            answer=from_english(DECLINE_MESSAGE, language),
            sources=sources,
            english_query=english_query,
            english_answer=DECLINE_MESSAGE,
            grounded=False,
            top_score=top_score,
        )

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
        grounded=True,
        top_score=top_score,
    )
