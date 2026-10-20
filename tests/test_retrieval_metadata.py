"""Retrieval keeps source metadata end to end, and says so when nothing is found."""

from app.agents.nodes import retriever
from app.services.retrieval import ranking_service


def _hit(i, source="pods.html", score=0.9):
    return {"content": f"chunk {i}", "source": source, "chunk_id": f"id-{i}", "score": score}


class FakeRanker:
    """Stands in for FlashRank: returns passages best-first with a score."""

    def rerank(self, request):
        passages = list(request.passages)[::-1]  # reverse, so ordering visibly changes
        return [{**p, "score": 1.0 - 0.1 * n} for n, p in enumerate(passages)]


def test_rerank_hits_keeps_metadata_and_reorders(monkeypatch):
    monkeypatch.setattr(ranking_service, "_get_ranker", lambda: FakeRanker())
    hits = [_hit(0, "a.html"), _hit(1, "b.html"), _hit(2, "c.html")]

    out = ranking_service.rerank_hits("q", hits, top_n=2)

    assert [h["chunk_id"] for h in out] == ["id-2", "id-1"]
    assert [h["source"] for h in out] == ["c.html", "b.html"]
    assert out[0]["rerank_score"] > out[1]["rerank_score"]


def test_rerank_hits_falls_back_to_original_order_on_error(monkeypatch):
    def boom():
        raise RuntimeError("model missing")

    monkeypatch.setattr(ranking_service, "_get_ranker", boom)
    hits = [_hit(0), _hit(1), _hit(2)]

    out = ranking_service.rerank_hits("q", hits, top_n=2)

    assert [h["chunk_id"] for h in out] == ["id-0", "id-1"]


def test_rerank_hits_empty_input():
    assert ranking_service.rerank_hits("q", [], top_n=5) == []


def _state(messages, rewritten="Kubernetes Job completions field meaning documentation"):
    return {"current_query": rewritten, "messages": messages, "plan": ["Start"]}


# The strings below are real planner outputs recorded in evals/results/agent-always.json.
def test_clean_search_query_removes_control_characters():
    assert retriever.clean_search_query("Kubernetes Secrets vs ConfigMaps differences guide\x00") == (
        "Kubernetes Secrets vs ConfigMaps differences guide"
    )


def test_clean_search_query_collapses_a_text_repeated_with_no_separator():
    doubled = "Kubernetes DaemonSet purpose and use casesKubernetes DaemonSet purpose and use cases"
    assert retriever.clean_search_query(doubled) == "Kubernetes DaemonSet purpose and use cases"


def test_clean_search_query_removes_web_search_operators():
    raw = (
        'Prometheus alert for Kubernetes CronJob not executed in over an hour '
        'site:kubernetes.io OR site:github.com "CronJob" "alert" "not run" "hour"'
    )
    assert retriever.clean_search_query(raw) == (
        'Prometheus alert for Kubernetes CronJob not executed in over an hour "CronJob" "alert" "not run" "hour"'
    )


def test_clean_search_query_handles_a_duplicate_that_also_has_operators():
    one = 'Kubernetes version that removed Docker site:kubernetes.io documentation'
    assert retriever.clean_search_query(one + one) == "Kubernetes version that removed Docker documentation"


def test_clean_search_query_leaves_good_queries_alone():
    for good in ["What is a CronJob?", "Kubernetes pause container purpose for a Pod", "go go go"]:
        assert retriever.clean_search_query(good) == good


def test_clean_search_query_does_not_collapse_short_repeats():
    assert retriever.clean_search_query("ha ha") == "ha ha"


def test_clean_search_query_can_return_empty():
    assert retriever.clean_search_query("\x00\x00") == ""


def test_empty_rewrite_on_a_follow_up_falls_back_to_the_users_words():
    history = [
        {"role": "user", "content": "What is a CronJob?"},
        {"role": "assistant", "content": "A wrapper around Jobs."},
        {"role": "user", "content": "How do I set its timezone?"},
    ]
    state = _state(history, rewritten="\x00")
    assert retriever.choose_search_query(state, "followups") == "How do I set its timezone?"


def test_follow_up_rewrite_is_cleaned_before_searching():
    history = [
        {"role": "user", "content": "What is a CronJob?"},
        {"role": "assistant", "content": "A wrapper around Jobs."},
        {"role": "user", "content": "How do I set its timezone?"},
    ]
    state = _state(history, rewritten="CronJob timeZone field\x00")
    assert retriever.choose_search_query(state, "followups") == "CronJob timeZone field"
    assert retriever.choose_search_query(state, "followups", clean=False) == "CronJob timeZone field\x00"


def test_first_question_searches_with_the_users_own_words():
    state = _state([{"role": "user", "content": "What does completions: 5 mean for a Job?"}])
    assert retriever.choose_search_query(state, "followups") == "What does completions: 5 mean for a Job?"


def test_follow_up_uses_the_planners_rewrite():
    history = [
        {"role": "user", "content": "What is a CronJob?"},
        {"role": "assistant", "content": "A wrapper around Jobs."},
        {"role": "user", "content": "How do I set its timezone?"},
    ]
    state = _state(history, rewritten="CronJob timeZone field")
    assert retriever.choose_search_query(state, "followups") == "CronJob timeZone field"


def test_always_mode_keeps_the_old_rewrite_behaviour():
    state = _state([{"role": "user", "content": "original question"}], rewritten="rewritten query")
    assert retriever.choose_search_query(state, "always") == "rewritten query"


def test_missing_messages_falls_back_to_the_rewrite():
    assert retriever.choose_search_query({"current_query": "only this"}, "followups") == "only this"


def test_retrieve_node_searches_with_the_chosen_query(monkeypatch):
    seen = {}

    def fake_search(query, limit):
        seen["query"] = query
        return [_hit(0)]

    monkeypatch.setattr(retriever, "search_enterprise_knowledge", fake_search)
    monkeypatch.setattr(retriever, "rerank_hits", lambda q, hits, top_n: hits[:top_n])
    monkeypatch.setattr(retriever.settings, "QUERY_REWRITE", "followups")

    retriever.retrieve_node(_state([{"role": "user", "content": "the user's words"}]))

    assert seen["query"] == "the user's words"


def test_retrieve_node_returns_citations(monkeypatch):
    monkeypatch.setattr(retriever, "search_enterprise_knowledge", lambda q, limit: [_hit(0), _hit(1)])
    monkeypatch.setattr(retriever, "rerank_hits", lambda q, hits, top_n: hits[:top_n])

    out = retriever.retrieve_node({"current_query": "what is a pod", "plan": ["Start"]})

    assert out["documents"] == ["chunk 0", "chunk 1"]
    assert [c["source"] for c in out["citations"]] == ["pods.html", "pods.html"]
    assert [c["chunk_id"] for c in out["citations"]] == ["id-0", "id-1"]
    assert "Context Retrieved" in out["plan"]


def test_retrieve_node_reports_no_context(monkeypatch):
    monkeypatch.setattr(retriever, "search_enterprise_knowledge", lambda q, limit: [])
    monkeypatch.setattr(retriever, "rerank_hits", lambda q, hits, top_n: hits[:top_n])

    out = retriever.retrieve_node({"current_query": "obscure", "plan": ["Start"]})

    assert out["documents"] == []
    assert out["citations"] == []
    assert "No relevant" in out["status"]
    assert "No Context Found" in out["plan"]
