"""The eval runner wires the golden set to the metrics, without any live service."""

import json
from types import SimpleNamespace

import pytest

from evals import run_retrieval_eval as runner


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)


def _api_args():
    return SimpleNamespace(mode="api", k=5, no_rerank=False, label="t", url="http://backend")


def _good(sources=("a.txt",)):
    return {"ranked_sources": list(sources), "top_score": 0.9, "answer": "an answer", "failed": False, "search_term": None}


def _failed():
    return {"ranked_sources": [], "top_score": None, "answer": "I apologize, but I encountered an internal error", "failed": True, "search_term": None}


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
    assert report["complete"] is True

    monkeypatch.setattr(runner, "run_retriever", useless)
    assert runner.run(args)["summary"]["hit"]["mean"] == 0.0


def test_search_term_is_read_from_the_thought_process():
    data = {"thought_process": ["Intent: Technical", "Search Term: backoffLimit retries"]}
    assert runner._search_term(data) == "backoffLimit retries"
    assert runner._search_term({"thought_process": ["Intent: Conversational/Memory"]}) is None
    assert runner._search_term({}) is None


def test_api_error_status_is_flagged_as_failed_not_scored(monkeypatch):
    monkeypatch.setattr(runner, "_post_with_retry", lambda url, q, i: {"status": "error", "answer": "I apologize..."})
    outcome = runner.run_api("anything", 0, "http://backend")
    assert outcome["failed"] is True
    assert outcome["ranked_sources"] == []


def test_api_success_reads_citations_and_search_term(monkeypatch):
    data = {
        "status": "ok",
        "answer": "Use backoffLimit.",
        "citations": [{"source": "cronjobs.docx", "score": 0.4}, {"source": "monitor_job.docx", "score": 0.9}],
        "thought_process": ["Intent: Technical", "Search Term: job retries"],
    }
    monkeypatch.setattr(runner, "_post_with_retry", lambda url, q, i: data)
    outcome = runner.run_api("anything", 0, "http://backend")
    assert outcome["failed"] is False
    assert outcome["ranked_sources"] == ["cronjobs.docx", "monitor_job.docx"]
    assert outcome["top_score"] == 0.9
    assert outcome["search_term"] == "job retries"


def _fake_post(monkeypatch, responses):
    import requests

    calls = []

    def post(url, json=None, timeout=None):
        calls.append(url)
        data = responses[min(len(calls) - 1, len(responses) - 1)]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(requests, "post", post)
    return calls


def test_post_with_retry_retries_an_error_status_then_succeeds(monkeypatch):
    calls = _fake_post(monkeypatch, [{"status": "error"}, {"status": "ok", "answer": "fine"}])
    data = runner._post_with_retry("http://backend", "q", 0)
    assert data["answer"] == "fine"
    assert len(calls) == 2


def test_post_with_retry_gives_up_after_the_last_retry(monkeypatch):
    calls = _fake_post(monkeypatch, [{"status": "error", "answer": "quota"}])
    data = runner._post_with_retry("http://backend", "q", 0)
    assert data["status"] == "error"
    assert len(calls) == 1 + len(runner.RETRY_DELAYS)


def test_run_stops_after_consecutive_failures(monkeypatch):
    calls = []

    def always_fail(question, index, url):
        calls.append(question)
        return _failed()

    monkeypatch.setattr(runner, "run_api", always_fail)
    report = runner.run(_api_args())

    assert len(calls) == runner.MAX_CONSECUTIVE_FAILURES
    assert report["stopped_early"] is True
    assert report["complete"] is False
    assert report["summary"]["n_failed"] == runner.MAX_CONSECUTIVE_FAILURES
    assert report["summary"]["n_missing"] == len(runner.load_questions()) - runner.MAX_CONSECUTIVE_FAILURES
    assert report["summary"]["n_answerable"] == 0


def test_a_single_failure_does_not_stop_the_run(monkeypatch):
    seen = []

    def flaky(question, index, url):
        seen.append(question)
        return _failed() if len(seen) == 2 else _good()

    monkeypatch.setattr(runner, "run_api", flaky)
    report = runner.run(_api_args())

    assert len(seen) == len(runner.load_questions())
    assert report["stopped_early"] is False
    assert report["summary"]["n_failed"] == 1
    assert report["complete"] is False  # one question still failed


def test_resume_keeps_good_results_and_reruns_only_failed_ones(monkeypatch):
    questions = runner.load_questions()
    first, second = questions[0], questions[1]
    previous = [
        {"id": first["id"], "gold": runner.gold_sources(first), **_good(["kept.txt"])},
        {"id": second["id"], "gold": runner.gold_sources(second), **_failed()},
    ]
    asked = []

    def fresh(question, index, url):
        asked.append(question)
        return _good(["fresh.txt"])

    monkeypatch.setattr(runner, "run_api", fresh)
    report = runner.run(_api_args(), previous=previous)

    assert first["question"] not in asked
    assert second["question"] in asked
    by_id = {r["id"]: r for r in report["results"]}
    assert by_id[first["id"]]["ranked_sources"] == ["kept.txt"]
    assert by_id[second["id"]]["ranked_sources"] == ["fresh.txt"]
    assert report["complete"] is True


def _stub_agent(monkeypatch, planner_output, searched):
    """Replace the planner with a stub and Qdrant/FlashRank with stubs; record what was searched."""
    import sys

    from app.agents.nodes import retriever

    def planner(state):
        if isinstance(planner_output, Exception):
            raise planner_output
        return planner_output

    monkeypatch.setitem(sys.modules, "app.agents.nodes.planner", SimpleNamespace(planner_node=planner))

    def search(query, limit):
        searched.append(query)
        return [{"content": "text", "source": "cronjobs.docx", "chunk_id": "c1", "score": 0.5}]

    monkeypatch.setattr(retriever, "search_enterprise_knowledge", search)
    monkeypatch.setattr(retriever, "rerank_hits", lambda q, hits, top_n: [{**h, "rerank_score": 0.9} for h in hits][:top_n])


def test_agent_mode_searches_with_the_original_question_on_a_first_turn(monkeypatch):
    from app.config import settings

    searched = []
    _stub_agent(monkeypatch, {"current_query": "Kubernetes Job completions documentation", "plan": ["Intent: Technical"]}, searched)
    monkeypatch.setattr(settings, "QUERY_REWRITE", "followups")

    outcome = runner.run_agent("What does completions: 5 mean for a Job?")

    assert searched == ["What does completions: 5 mean for a Job?"]
    assert outcome["query_used"] == "What does completions: 5 mean for a Job?"
    assert outcome["search_term"] == "Kubernetes Job completions documentation"
    assert outcome["ranked_sources"] == ["cronjobs.docx"]
    assert outcome["intent"] == "technical" and outcome["failed"] is False


def test_agent_mode_always_rewrite_reproduces_the_old_behaviour(monkeypatch):
    from app.config import settings

    searched = []
    _stub_agent(monkeypatch, {"current_query": "Kubernetes Job completions documentation", "plan": ["Intent: Technical"]}, searched)
    monkeypatch.setattr(settings, "QUERY_REWRITE", "always")

    runner.run_agent("What does completions: 5 mean for a Job?")

    assert searched == ["Kubernetes Job completions documentation"]


def test_agent_mode_skips_retrieval_when_the_planner_says_conversational(monkeypatch):
    searched = []
    _stub_agent(monkeypatch, {"current_query": "CONVERSATIONAL", "plan": ["Intent: Conversational/Memory"]}, searched)

    outcome = runner.run_agent("hello")

    assert searched == []
    assert outcome["intent"] == "conversational"
    assert outcome["ranked_sources"] == []
    assert outcome["failed"] is False


def test_agent_mode_marks_a_planner_error_as_failed(monkeypatch):
    searched = []
    _stub_agent(monkeypatch, RuntimeError("429 rate limit"), searched)

    outcome = runner.run_agent("anything")

    assert outcome["failed"] is True
    assert "429" in outcome["error"]
    assert searched == []


def test_run_in_agent_mode_applies_the_rewrite_setting_and_uses_run_agent(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "QUERY_REWRITE", "followups")
    seen = []

    def fake_agent(question):
        seen.append(settings.QUERY_REWRITE)
        return _good()

    monkeypatch.setattr(runner, "run_agent", fake_agent)
    args = SimpleNamespace(mode="agent", k=5, no_rerank=False, label="t", url="", rewrite="always")

    report = runner.run(args)

    assert set(seen) == {"always"}
    assert report["config"]["rewrite"] == "always"
    assert report["config"]["mode"] == "agent"
    assert report["complete"] is True


def test_report_is_saved_after_every_question(monkeypatch, tmp_path):
    out = tmp_path / "run.json"
    calls = []

    def dies_on_third(question, index, url):
        calls.append(question)
        if len(calls) == 3:
            raise RuntimeError("backend went away")
        return _good()

    monkeypatch.setattr(runner, "run_api", dies_on_third)
    with pytest.raises(RuntimeError):
        runner.run(_api_args(), out_path=out)

    saved = json.loads(out.read_text())
    assert len(saved["results"]) == 2
    assert saved["complete"] is False
