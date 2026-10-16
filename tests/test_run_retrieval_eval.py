"""The eval runner wires the golden set to the metrics, without any live service."""

from types import SimpleNamespace

from evals import run_retrieval_eval as runner


def test_load_questions_returns_answerable_then_unanswerable():
    questions = runner.load_questions()
    ids = [q["id"] for q in questions]
    assert ids[0] == "Q001"
    assert any(i.startswith("U") for i in ids)
    assert all(q["question"].strip() for q in questions)


def test_gold_sources_are_unique_and_sorted():
    multi = next(q for q in runner.load_questions() if q["type"] == "multi_source")
    sources = runner.gold_sources(multi)
    assert sources == sorted(set(sources))
    assert len(sources) >= 2


def test_unanswerable_questions_have_no_gold_sources():
    unanswerable = [q for q in runner.load_questions() if q["id"].startswith("U")]
    assert unanswerable
    assert all(runner.gold_sources(q) == [] for q in unanswerable)


def test_run_scores_a_perfect_and_a_useless_retriever(monkeypatch):
    questions = runner.load_questions()
    gold_by_question = {q["question"]: runner.gold_sources(q) for q in questions}

    def perfect(question, k, rerank):
        gold = gold_by_question[question]
        return {"ranked_sources": (gold or ["unrelated.txt"]) + ["other.txt"], "top_score": 1.0 if gold else 0.1, "answer": None}

    def useless(question, k, rerank):
        return {"ranked_sources": ["unrelated.txt"] * k, "top_score": 0.0, "answer": None}

    args = SimpleNamespace(mode="retriever", k=5, no_rerank=False, label="t", url="")

    monkeypatch.setattr(runner, "run_retriever", perfect)
    report = runner.run(args)
    assert report["summary"]["hit"]["mean"] == 1.0
    assert report["summary"]["mrr"]["mean"] == 1.0
    assert report["summary"]["suggested_score_threshold"]["balanced_accuracy"] == 1.0
    assert len(report["results"]) == len(questions)
    assert report["label"] == "t"

    monkeypatch.setattr(runner, "run_retriever", useless)
    assert runner.run(args)["summary"]["hit"]["mean"] == 0.0
