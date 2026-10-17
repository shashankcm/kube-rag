"""The v2 golden set must stay grounded in the source documents."""

import json

import pytest

from evals import build_golden_v2 as g

pytestmark = pytest.mark.skipif(
    not g.PARSED_DIR.exists(), reason="processed_data/true is not available"
)


@pytest.fixture(scope="module")
def dataset():
    return json.loads(g.OUT_PATH.read_text())


def test_committed_file_matches_the_builder(dataset):
    assert dataset == g.build()


def test_every_evidence_span_appears_in_its_source(dataset):
    assert g.unverified_spans(dataset["questions"], g.load_corpus()) == []


def test_ids_are_unique(dataset):
    ids = [q["id"] for q in dataset["questions"] + dataset["unanswerable"]]
    assert len(ids) == len(set(ids))


def test_every_answerable_question_has_a_reference_and_evidence(dataset):
    for q in dataset["questions"]:
        assert q["reference"].strip(), q["id"]
        assert q["evidence"], q["id"]


def test_unanswerable_questions_have_no_evidence(dataset):
    assert all(q["evidence"] == [] for q in dataset["unanswerable"])


def test_set_is_large_enough_to_report_metrics(dataset):
    assert len(dataset["questions"]) >= 60
    assert len(dataset["unanswerable"]) >= 10
