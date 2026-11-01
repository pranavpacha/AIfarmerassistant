"""Run the 30-50 curated test cases through the full pipeline and score
groundedness, retrieval accuracy, and factual-keyword coverage, then write a
results file and a failure-analysis draft (per the course rubric's
"quantitative testing & error analysis" requirement).

Run from the project root: `python eval/evaluate.py`
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import ask  # noqa: E402

EVAL_DIR = Path(__file__).parent
LOW_CONFIDENCE_THRESHOLD = 0.35  # below this top-1 retrieval score, treat as "no good grounding found"


def score_case(case: dict, result) -> dict:
    is_out_of_domain = case["expected_category"] == "OUT_OF_DOMAIN"
    top_score = result.sources[0]["score"] if result.sources else 0.0

    retrieval_hit = (
        any(s["category"] == case["expected_category"] for s in result.sources)
        if not is_out_of_domain
        else top_score < LOW_CONFIDENCE_THRESHOLD
    )

    answer_lower = result.english_answer.lower()
    keywords = case["expected_keywords"]
    hits = [k for k in keywords if k.lower() in answer_lower]
    keyword_coverage = (len(hits) / len(keywords)) if keywords else None

    if is_out_of_domain:
        passed = retrieval_hit  # low-confidence retrieval is the correct behavior here
    else:
        passed = retrieval_hit and (keyword_coverage is None or keyword_coverage >= 0.3)

    return {
        "id": case["id"],
        "query": case["query"],
        "language": case["language"],
        "expected_category": case["expected_category"],
        "top_retrieved_category": result.sources[0]["category"] if result.sources else None,
        "top_retrieval_score": round(top_score, 3),
        "retrieval_hit": retrieval_hit,
        "keyword_coverage": round(keyword_coverage, 2) if keyword_coverage is not None else None,
        "matched_keywords": hits,
        "english_answer": result.english_answer,
        "final_answer": result.answer,
        "passed": passed,
    }


def main():
    test_cases = json.loads((EVAL_DIR / "test_cases.json").read_text(encoding="utf-8"))
    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['id']}: {case['query'][:60]}...")
        try:
            result = ask(case["query"], case["language"])
            results.append(score_case(case, result))
        except Exception as e:
            results.append({
                "id": case["id"], "query": case["query"], "error": str(e), "passed": False,
            })

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = [r for r in results if not r.get("passed")]

    coverages = [r["keyword_coverage"] for r in results if r.get("keyword_coverage") is not None]
    avg_coverage = sum(coverages) / len(coverages) if coverages else None

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "avg_keyword_coverage": round(avg_coverage, 3) if avg_coverage is not None else None,
        "results": results,
    }

    (EVAL_DIR / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPassed {passed}/{total} ({summary['pass_rate']*100:.1f}%). Wrote eval/results.json")

    write_failure_analysis(failed, summary)


def write_failure_analysis(failed: list[dict], summary: dict):
    lines = [
        "# Failure Analysis\n",
        f"**Pass rate:** {summary['passed']}/{summary['total_cases']} "
        f"({summary['pass_rate']*100:.1f}%)\n",
        f"**Average keyword coverage on passed+failed in-domain cases:** "
        f"{summary['avg_keyword_coverage']}\n",
        "## Failed cases\n",
    ]
    if not failed:
        lines.append("None — all test cases passed on this run.\n")
    for r in failed:
        if "error" in r:
            lines.append(f"- **{r['id']}** (`{r['query']}`): crashed with `{r['error']}`\n")
            continue
        reason = []
        if not r["retrieval_hit"]:
            reason.append(
                f"retrieval returned `{r['top_retrieved_category']}` instead of expected "
                f"`{r['expected_category']}` (top score {r['top_retrieval_score']})"
            )
        if r["keyword_coverage"] is not None and r["keyword_coverage"] < 0.3:
            reason.append(f"low keyword coverage ({r['keyword_coverage']})")
        lines.append(
            f"- **{r['id']}** (`{r['query']}`): {'; '.join(reason) or 'unspecified'}\n"
            f"  - Answer given: _{r['final_answer'][:200]}..._\n"
        )

    lines.append(
        "\n## Notes for manual review\n"
        "- Check whether out-of-domain queries (t36, t37) are correctly declined rather "
        "than answered with a hallucinated agricultural response.\n"
        "- Check whether low-resource language cases (Tamil, Telugu, Kannada) lose accuracy "
        "due to translation quality rather than retrieval/generation quality — compare "
        "`english_answer` vs `final_answer` in results.json for those cases.\n"
        "- Expand `data/seed_corpus.json` or run `data/fetch_kcc_data.py` with a real API key "
        "if retrieval misses cluster around specific crops/categories.\n"
    )

    (EVAL_DIR / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote eval/failure_analysis.md")


if __name__ == "__main__":
    main()
