"""Re-scoring saved results against corrected gold labels."""

from evals.rescore import rescore


def _report(results):
    return {"label": "t", "summary": {"k": 5, "n_missing": 0}, "results": results}


def _questions(*pairs):
    return [{"id": qid, "question": "q", "evidence": [{"source": s, "span": "x"} for s in sources]}
            for qid, sources in pairs]


def _record(qid, gold, ranked, **extra):
    return {"id": qid, "gold": gold, "ranked_sources": ranked, "top_score": None, "answer": None, **extra}


def test_a_miss_becomes_a_hit_when_the_retrieved_source_is_added_to_gold():
    report = _report([_record("Q1", ["a.txt"], ["b.txt", "a.txt"])])
    # Before: gold is a.txt, found at rank 2. After the correction gold gains b.txt, found at rank 1.
    updated = rescore(report, _questions(("Q1", ["a.txt", "b.txt"])))

    assert updated["results"][0]["gold"] == ["a.txt", "b.txt"]
    assert updated["summary"]["mrr"]["mean"] == 1.0
    assert updated["summary"]["recall"]["mean"] == 1.0
    assert updated["rescored"] is True


def test_the_original_report_is_not_modified():
    report = _report([_record("Q1", ["a.txt"], ["b.txt"])])
    rescore(report, _questions(("Q1", ["b.txt"])))
    assert report["results"][0]["gold"] == ["a.txt"]
    assert "rescored" not in report


def test_failed_records_stay_failed_and_out_of_the_metrics():
    report = _report([
        _record("Q1", ["a.txt"], ["a.txt"]),
        _record("Q2", ["a.txt"], [], failed=True),
    ])
    updated = rescore(report, _questions(("Q1", ["a.txt"]), ("Q2", ["a.txt"])))
    assert updated["summary"]["n_failed"] == 1
    assert updated["summary"]["n_answerable"] == 1


def test_unanswerable_questions_keep_empty_gold():
    report = _report([_record("U1", [], ["a.txt"], answer="I don't have enough information.")])
    updated = rescore(report, _questions(("U1", [])))
    assert updated["summary"]["n_unanswerable"] == 1
    assert updated["summary"]["refusal_rate_on_unanswerable"]["mean"] == 1.0
