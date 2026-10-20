"""Paired comparison of two eval runs."""

import pytest

from evals.compare import paired_difference


def _rec(qid, ranked, gold=("a.txt",), **extra):
    return {"id": qid, "gold": list(gold), "ranked_sources": ranked, "top_score": None, "answer": None, **extra}


def _report(*records):
    return {"results": list(records)}


def test_identical_runs_have_zero_difference_and_a_zero_width_interval():
    run = _report(_rec("Q1", ["a.txt"]), _rec("Q2", ["x.txt"]), _rec("Q3", ["a.txt"]))
    result = paired_difference(run, run)
    assert result["n"] == 3
    for metric in ("hit", "mrr"):
        assert result[metric]["diff"] == 0.0
        assert result[metric]["ci95"] == [0.0, 0.0]
        assert result[metric]["same"] == 3


def test_a_better_than_b_gives_a_positive_difference_with_counts():
    better = _report(_rec("Q1", ["a.txt"]), _rec("Q2", ["a.txt"]), _rec("Q3", ["x.txt", "a.txt"]))
    worse = _report(_rec("Q1", ["x.txt"]), _rec("Q2", ["a.txt"]), _rec("Q3", ["a.txt"]))
    result = paired_difference(better, worse)

    # hit: Q1 better (+1), Q2 same, Q3 same -> mean +1/3
    assert result["hit"]["diff"] == pytest.approx(1 / 3)
    assert (result["hit"]["better"], result["hit"]["worse"], result["hit"]["same"]) == (1, 0, 2)
    # mrr: Q1 +1.0, Q2 0, Q3 0.5 - 1.0 = -0.5 -> mean +0.5/3
    assert result["mrr"]["diff"] == pytest.approx(0.5 / 3)
    assert (result["mrr"]["better"], result["mrr"]["worse"], result["mrr"]["same"]) == (1, 1, 1)


def test_only_questions_scored_by_both_runs_and_answerable_are_compared():
    a = _report(_rec("Q1", ["a.txt"]), _rec("Q2", ["a.txt"], failed=True), _rec("U1", ["a.txt"], gold=()))
    b = _report(_rec("Q1", ["a.txt"]), _rec("Q2", ["a.txt"]), _rec("Q3", ["a.txt"]))
    assert paired_difference(a, b)["n"] == 1


def test_the_bootstrap_is_deterministic():
    a = _report(*[_rec(f"Q{i}", ["a.txt"] if i % 2 else ["x.txt"]) for i in range(10)])
    b = _report(*[_rec(f"Q{i}", ["x.txt"]) for i in range(10)])
    assert paired_difference(a, b) == paired_difference(a, b)


def test_the_interval_contains_the_mean():
    a = _report(*[_rec(f"Q{i}", ["a.txt"] if i % 3 else ["x.txt"]) for i in range(12)])
    b = _report(*[_rec(f"Q{i}", ["x.txt"]) for i in range(12)])
    m = paired_difference(a, b)["hit"]
    assert m["ci95"][0] <= m["diff"] <= m["ci95"][1]
