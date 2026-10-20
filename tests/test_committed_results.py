"""
Guards the evaluation results that are committed to the repo.

CI cannot re-run retrieval (it has no Qdrant or API keys), so these checks protect what it can:
the saved results must agree with the current gold labels, and the headline numbers must not
silently fall below a conservative floor.
"""

import json

import pytest

from evals.rescore import rescore
from evals.run_retrieval_eval import RESULTS_DIR, load_questions

# Skip local backup copies such as baseline-api.before-failed-flag.json.
RESULT_FILES = sorted(p for p in RESULTS_DIR.glob("*.json") if ".before-" not in p.name)


def _load(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def test_there_are_saved_results_to_check():
    assert RESULT_FILES


@pytest.mark.parametrize("path", RESULT_FILES, ids=lambda p: p.stem)
def test_saved_results_were_scored_against_the_current_gold_labels(path):
    report = json.loads(path.read_text())
    questions = {q["id"]: q for q in load_questions()}
    stored = {r["id"]: sorted(r["gold"]) for r in report["results"]}
    current = rescore(report, list(questions.values()))
    expected = {r["id"]: sorted(r["gold"]) for r in current["results"]}
    assert stored == expected, f"{path.name} has stale gold labels; run python -m evals.rescore {path.stem} --write"


@pytest.mark.parametrize("path", RESULT_FILES, ids=lambda p: p.stem)
def test_saved_summaries_match_their_own_results(path):
    report = json.loads(path.read_text())
    recomputed = rescore(report, load_questions())["summary"]
    for metric in ("hit", "recall", "precision", "mrr"):
        assert report["summary"][metric]["mean"] == pytest.approx(recomputed[metric]["mean"]), metric


@pytest.mark.parametrize("label", ["baseline-dense", "agent-followups"])
def test_headline_retrieval_results_stay_above_a_floor(label):
    summary = _load(label)["summary"]
    assert summary["hit"]["mean"] >= 0.95
    assert summary["mrr"]["mean"] >= 0.90


def test_the_reranker_is_still_shown_to_help():
    with_rerank = _load("baseline-dense")["summary"]
    without = _load("baseline-dense-norerank")["summary"]
    assert with_rerank["mrr"]["mean"] > without["mrr"]["mean"]
    assert with_rerank["hit"]["mean"] > without["hit"]["mean"]


def test_the_complete_api_run_refuses_most_unanswerable_questions():
    report = _load("baseline-api")
    assert report["complete"] is True
    assert report["summary"]["n_failed"] == 0
    assert report["summary"]["refusal_rate_on_unanswerable"]["mean"] >= 0.8
