"""
Paired comparison of two saved eval runs.

Both runs score the same questions, so the per-question difference removes question-to-question
noise and is more sensitive than comparing two separate confidence intervals. The interval is a
percentile bootstrap over questions.

  python -m evals.compare agent-followups agent-always     # B minus A is printed as "A minus B"
"""

import argparse
import json
import random
from statistics import mean

from evals.retrieval_metrics import question_metrics
from evals.run_retrieval_eval import RESULTS_DIR

METRICS = ("hit", "recall", "precision", "mrr")


def paired_difference(report_a: dict, report_b: dict, k: int = 5, n_resamples: int = 5000, seed: int = 0) -> dict:
    """
    A minus B for each metric over the answerable questions that both runs scored (neither failed).
    Returns {metric: {"diff", "ci95", "better", "worse", "same"}} plus "n".
    """
    a = {r["id"]: r for r in report_a["results"] if r["gold"] and not r.get("failed")}
    b = {r["id"]: r for r in report_b["results"] if r["gold"] and not r.get("failed")}
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    out = {"n": len(ids)}
    for metric in METRICS:
        diffs = [
            question_metrics(a[i]["ranked_sources"], a[i]["gold"], k)[metric]
            - question_metrics(b[i]["ranked_sources"], b[i]["gold"], k)[metric]
            for i in ids
        ]
        means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_resamples)) if diffs else [0.0]
        out[metric] = {
            "diff": mean(diffs) if diffs else 0.0,
            "ci95": [means[int(0.025 * len(means))], means[min(len(means) - 1, int(0.975 * len(means)))]],
            "better": sum(d > 0 for d in diffs),
            "worse": sum(d < 0 for d in diffs),
            "same": sum(d == 0 for d in diffs),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="label of run A in evals/results/")
    parser.add_argument("b", help="label of run B in evals/results/")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    result = paired_difference(
        json.loads((RESULTS_DIR / f"{args.a}.json").read_text()),
        json.loads((RESULTS_DIR / f"{args.b}.json").read_text()),
        args.k,
    )
    print(f"{args.a} minus {args.b}, {result['n']} answerable questions scored by both, k={args.k}")
    for metric in METRICS:
        m = result[metric]
        lo, hi = m["ci95"]
        print(f"  {metric:9} {m['diff']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   "
              f"A better/worse/same: {m['better']}/{m['worse']}/{m['same']}")


if __name__ == "__main__":
    main()
