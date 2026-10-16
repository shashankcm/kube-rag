"""
Retrieval metrics that need no LLM judge.

Gold labels are source files (see evals/golden_v2.json). For each question we
look at the ranked list of sources behind the retrieved chunks and compute:

- hit@k        did any gold source appear in the top k chunks
- recall@k     share of the gold sources that appear in the top k chunks
- precision@k  share of the top k chunks that come from a gold source
- mrr@k        1 / rank of the first chunk from a gold source (0 if none)

Unanswerable questions are scored by how often the system refuses, and their
top reranker scores are used to suggest a relevance threshold.

Every aggregate carries a 95% bootstrap confidence interval over questions,
so a difference between two retrievers can be judged against the noise.
"""

import random
from statistics import mean

REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "i don't know",
    "i do not know",
)


def question_metrics(ranked_sources: list[str], gold_sources: list[str], k: int) -> dict:
    """Metrics for one answerable question. `ranked_sources` has one entry per chunk, best first."""
    gold = set(gold_sources)
    top = ranked_sources[:k]
    found = gold & set(top)
    first_rank = next((i for i, s in enumerate(top, 1) if s in gold), None)
    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(gold) if gold else 0.0,
        "precision": sum(s in gold for s in top) / k if k else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap interval for the mean. Returns (low, high)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = sorted(mean(rng.choices(values, k=size)) for _ in range(n_resamples))
    low = means[int((alpha / 2) * n_resamples)]
    high = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (low, high)


def is_refusal(answer: str | None) -> bool:
    text = (answer or "").lower().replace("’", "'")
    return any(marker in text for marker in REFUSAL_MARKERS)


def best_threshold(answerable_scores: list[float], unanswerable_scores: list[float]):
    """
    Pick the score cut-off that best separates answerable from unanswerable
    questions (balanced accuracy). A question is treated as answerable when its
    top score is >= the threshold. Returns (threshold, balanced_accuracy), or
    None when either list is empty.
    """
    if not answerable_scores or not unanswerable_scores:
        return None
    best = None
    for t in sorted(set(answerable_scores) | set(unanswerable_scores)):
        tpr = sum(s >= t for s in answerable_scores) / len(answerable_scores)
        tnr = sum(s < t for s in unanswerable_scores) / len(unanswerable_scores)
        balanced = (tpr + tnr) / 2
        if best is None or balanced > best[1]:
            best = (t, balanced)
    return best


def summarize(results: list[dict], k: int) -> dict:
    """
    `results` holds one record per question:
      {"id", "gold": [sources] (empty for unanswerable), "ranked_sources": [...],
       "top_score": float | None, "answer": str | None}
    """
    answerable = [r for r in results if r["gold"]]
    unanswerable = [r for r in results if not r["gold"]]

    per_question = [question_metrics(r["ranked_sources"], r["gold"], k) for r in answerable]
    summary = {"k": k, "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}
    for name in ("hit", "recall", "precision", "mrr"):
        values = [m[name] for m in per_question]
        low, high = bootstrap_ci(values)
        summary[name] = {"mean": mean(values) if values else 0.0, "ci95": [low, high]}

    answered = [r for r in unanswerable if r.get("answer") is not None]
    if answered:
        refusals = [1.0 if is_refusal(r["answer"]) else 0.0 for r in answered]
        low, high = bootstrap_ci(refusals)
        summary["refusal_rate_on_unanswerable"] = {"mean": mean(refusals), "ci95": [low, high], "n": len(refusals)}

    pos = [r["top_score"] for r in answerable if r.get("top_score") is not None]
    neg = [r["top_score"] for r in unanswerable if r.get("top_score") is not None]
    cut = best_threshold(pos, neg)
    if cut:
        summary["suggested_score_threshold"] = {"threshold": cut[0], "balanced_accuracy": cut[1]}
    return summary
