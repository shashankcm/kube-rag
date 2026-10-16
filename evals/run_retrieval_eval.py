"""
Runs the v2 golden set against kube-rag and writes retrieval metrics.

Two modes:

  retriever  Calls Qdrant and the reranker directly. No LLM, so it needs only
             Qdrant and the embedding model. Use this for the baseline and for
             comparing retrieval changes. It also records each question's top
             reranker score, which is used to suggest a relevance threshold.

  api        Calls the running FastAPI /query endpoint. Slower (rate-limited
             LLM calls) but also scores how often unanswerable questions are
             refused.

Examples:
  python -m evals.run_retrieval_eval --label baseline-dense --mode retriever
  python -m evals.run_retrieval_eval --label baseline-dense-norerank --mode retriever --no-rerank
  python -m evals.run_retrieval_eval --label baseline-api --mode api
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.retrieval_metrics import summarize

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_v2.json"
RESULTS_DIR = EVALS_DIR / "results"

CANDIDATES = 15           # same candidate pool the agent uses before reranking
API_DELAY_SECONDS = 10    # stays within the Groq requests-per-minute limit
API_TIMEOUT_SECONDS = 120


def load_questions(path: Path = GOLDEN_PATH) -> list[dict]:
    data = json.loads(path.read_text())
    return data["questions"] + data["unanswerable"]


def gold_sources(question: dict) -> list[str]:
    return sorted({e["source"] for e in question["evidence"]})


def run_retriever(question: str, k: int, rerank: bool) -> dict:
    # Imported lazily so the metrics and tests do not need Qdrant or the models.
    from app.services.retrieval.qdrant_service import search_enterprise_knowledge
    from app.services.retrieval.ranking_service import rerank_hits

    hits = search_enterprise_knowledge(question, limit=CANDIDATES)
    ranked = rerank_hits(question, hits, top_n=k) if rerank else hits[:k]
    top = ranked[0] if ranked else None
    score = None
    if top:
        score = top.get("rerank_score", top.get("score"))
    return {
        "ranked_sources": [h["source"] for h in ranked],
        "top_score": score,
        "answer": None,
    }


def run_api(question: str, index: int, url: str) -> dict:
    import requests

    resp = requests.post(
        f"{url.rstrip('/')}/query",
        json={"q": question, "thread_id": f"retrieval_eval_{index}"},
        timeout=API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    citations = data.get("citations") or []
    scores = [c["score"] for c in citations if c.get("score") is not None]
    return {
        "ranked_sources": [c["source"] for c in citations],
        "top_score": max(scores) if scores else None,
        "answer": data.get("answer") or "",
    }


def run(args) -> dict:
    questions = load_questions()
    results = []
    for i, q in enumerate(questions):
        if args.mode == "retriever":
            outcome = run_retriever(q["question"], args.k, rerank=not args.no_rerank)
        else:
            outcome = run_api(q["question"], i, args.url)
            time.sleep(API_DELAY_SECONDS)
        results.append({"id": q["id"], "gold": gold_sources(q), **outcome})
        print(f"[{i + 1}/{len(questions)}] {q['id']}", flush=True)

    config = {"mode": args.mode, "k": args.k, "candidates": CANDIDATES, "rerank": not args.no_rerank}
    if args.mode == "retriever":
        from app.config import settings

        config["embedding_model"] = settings.EMBEDDING_MODEL
        config["collection"] = settings.QDRANT_COLLECTION
    return {
        "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "summary": summarize(results, args.k),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="name for this run, used as the output file name")
    parser.add_argument("--mode", choices=["retriever", "api"], default="retriever")
    parser.add_argument("--k", type=int, default=5, help="rank cut-off for the metrics")
    parser.add_argument("--no-rerank", action="store_true", help="retriever mode: skip FlashRank")
    parser.add_argument("--url", default="http://localhost:8000", help="api mode: backend base URL")
    args = parser.parse_args()

    report = run(args)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    s = report["summary"]
    print(f"\nwrote {out.relative_to(EVALS_DIR.parent)}")
    for name in ("hit", "recall", "precision", "mrr"):
        lo, hi = s[name]["ci95"]
        print(f"  {name}@{s['k']:<2} {s[name]['mean']:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    if "refusal_rate_on_unanswerable" in s:
        r = s["refusal_rate_on_unanswerable"]
        print(f"  refusal rate on unanswerable: {r['mean']:.2f} (n={r['n']})")
    if "suggested_score_threshold" in s:
        t = s["suggested_score_threshold"]
        print(f"  suggested score threshold: {t['threshold']:.3f} (balanced accuracy {t['balanced_accuracy']:.2f})")


if __name__ == "__main__":
    main()
