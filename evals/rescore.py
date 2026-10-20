"""
Re-scores saved eval results against the current golden labels, without re-running anything.

Each saved result keeps the sources the system retrieved, so when a gold label is corrected
(see the changelog in evals/golden_v2.json) the old runs can be scored again offline.

  python -m evals.rescore baseline-dense agent-followups          # print before and after
  python -m evals.rescore baseline-dense agent-followups --write  # also update the files
"""

import argparse
import json

from evals.retrieval_metrics import summarize
from evals.run_retrieval_eval import RESULTS_DIR, gold_sources, load_questions


def rescore(report: dict, questions: list[dict]) -> dict:
    """Return a copy of `report` with every record's gold replaced by the current labels."""
    gold = {q["id"]: gold_sources(q) for q in questions}
    results = [{**r, "gold": gold[r["id"]]} for r in report["results"]]
    summary = summarize(results, report["summary"]["k"])
    summary["n_missing"] = report["summary"].get("n_missing", 0)
    return {**report, "results": results, "summary": summary, "rescored": True}


def _line(summary: dict) -> str:
    return (f"hit@{summary['k']} {summary['hit']['mean']:.3f}   "
            f"recall {summary['recall']['mean']:.3f}   mrr {summary['mrr']['mean']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+", help="result files in evals/results/, without .json")
    parser.add_argument("--write", action="store_true", help="update the result files in place")
    args = parser.parse_args()

    questions = load_questions()
    for label in args.labels:
        path = RESULTS_DIR / f"{label}.json"
        report = json.loads(path.read_text())
        updated = rescore(report, questions)
        print(f"{label}\n  before: {_line(report['summary'])}\n  after : {_line(updated['summary'])}")
        if args.write:
            path.write_text(json.dumps(updated, indent=2) + "\n")


if __name__ == "__main__":
    main()
