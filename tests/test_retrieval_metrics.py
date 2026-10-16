"""Hand-computed checks for the LLM-free retrieval metrics."""

import pytest

from evals.retrieval_metrics import (
    best_threshold,
    bootstrap_ci,
    is_refusal,
    question_metrics,
    summarize,
)


def test_question_metrics_partial_hit():
    # top 3 = [a, b, a]; gold = {a, c}; only "a" is found
    m = question_metrics(["a", "b", "a", "c"], ["a", "c"], k=3)
    assert m["hit"] == 1.0
    assert m["recall"] == 0.5
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["mrr"] == 1.0


def test_question_metrics_miss():
    m = question_metrics(["a", "b"], ["z"], k=2)
    assert m == {"hit": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0}


def test_mrr_depends_on_rank_and_respects_k():
    ranked = ["x", "y", "gold"]
    assert question_metrics(ranked, ["gold"], k=3)["mrr"] == pytest.approx(1 / 3)
    assert question_metrics(ranked, ["gold"], k=2)["mrr"] == 0.0


def test_bootstrap_ci_edge_cases_and_bounds():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.7, 0.7, 0.7]) == (0.7, 0.7)
    low, high = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    assert 0.0 <= low <= 0.625 <= high <= 1.0


def test_bootstrap_ci_is_deterministic():
    values = [0.0, 1.0, 1.0, 0.5, 0.25]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_is_refusal_handles_curly_apostrophes_and_none():
    assert is_refusal("I don’t have enough information in the documentation.")
    assert is_refusal("Sorry, I don't know.")
    assert not is_refusal("A Pod is the smallest deployable unit.")
    assert not is_refusal(None)


def test_best_threshold_on_separable_scores():
    assert best_threshold([0.8, 0.9], [0.1, 0.2]) == (0.8, 1.0)


def test_best_threshold_needs_both_groups():
    assert best_threshold([], [0.1]) is None
    assert best_threshold([0.9], []) is None


def _record(qid, gold, ranked, score=None, answer=None):
    return {"id": qid, "gold": gold, "ranked_sources": ranked, "top_score": score, "answer": answer}


def test_summarize_separates_answerable_from_unanswerable():
    results = [
        _record("Q1", ["a"], ["a", "x"], score=0.9),
        _record("Q2", ["b"], ["x", "y", "z"], score=0.2),
        _record("U1", [], ["x"], score=0.1, answer="I don't have enough information."),
        _record("U2", [], ["y"], score=0.3, answer="Pods are..."),
    ]
    s = summarize(results, k=3)

    assert (s["k"], s["n_answerable"], s["n_unanswerable"]) == (3, 2, 2)
    assert s["hit"]["mean"] == 0.5
    assert s["mrr"]["mean"] == 0.5
    assert s["hit"]["ci95"][0] <= 0.5 <= s["hit"]["ci95"][1]
    assert s["refusal_rate_on_unanswerable"]["mean"] == 0.5
    assert s["refusal_rate_on_unanswerable"]["n"] == 2
    assert "suggested_score_threshold" in s


def test_summarize_without_answers_or_scores_omits_those_sections():
    s = summarize([_record("Q1", ["a"], ["a"]), _record("U1", [], ["x"])], k=1)
    assert "refusal_rate_on_unanswerable" not in s
    assert "suggested_score_threshold" not in s
