"""
Runs the v2 golden set against kube-rag and writes retrieval metrics.

Two modes:

  retriever  Calls Qdrant and the reranker directly. No LLM, so it needs only
             Qdrant and the embedding model. Use this for the baseline and for
             comparing retrieval changes. It also records each question's top
             reranker score, which is used to suggest a relevance threshold.

  agent      Runs the real planner and the real retriever, with no guardrails
             and no answer generation. Isolates the planner's query rewriting
             without using the guardrail model's daily token quota.
             --rewrite always|followups switches the old and new behaviour.

  api        Calls the running FastAPI /query endpoint. Slower (rate-limited
             LLM calls) but also scores how often unanswerable questions are
             refused.

The results file is saved after every question. A failed request (the API
reports errors as HTTP 200 with status "error") is recorded as failed and left
out of the metrics. After MAX_CONSECUTIVE_FAILURES failed questions in a row the
run stops, which is what a daily LLM quota looks like. Re-run with --resume to
keep the good results and redo only the failed or missing questions.

Examples:
  python -m evals.run_retrieval_eval --label baseline-dense --mode retriever
  python -m evals.run_retrieval_eval --label baseline-dense-norerank --mode retriever --no-rerank
  python -m evals.run_retrieval_eval --label baseline-api --mode api
  python -m evals.run_retrieval_eval --label baseline-api --mode api --resume
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

CANDIDATES = 15                # same candidate pool the agent uses before reranking
API_DELAY_SECONDS = 10         # stays within the Groq requests-per-minute limit
AGENT_DELAY_SECONDS = 2        # the planner makes one short LLM call per question
API_TIMEOUT_SECONDS = 120
RETRY_DELAYS = (15, 45)        # seconds to wait before the 1st and 2nd retry
MAX_CONSECUTIVE_FAILURES = 3   # then stop; a daily quota will not clear in minutes


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
        "failed": False,
        "search_term": None,
    }


def run_agent(question: str) -> dict:
    """
    The real planner and the real retriever, with no guardrails and no answer generation.

    This isolates the effect of the planner's query rewriting without spending the
    guardrail model's daily token quota. Which query reaches the retriever depends on
    settings.QUERY_REWRITE ("always" or "followups").
    """
    from app.agents.nodes.planner import planner_node
    from app.agents.nodes.retriever import choose_search_query, retrieve_node
    from app.config import settings

    state = {
        "messages": [{"role": "user", "content": question}],
        "current_query": question,
        "documents": [],
        "citations": [],
        "plan": ["Start"],
        "status": "",
    }
    try:
        state.update(planner_node(state))
    except Exception as exc:  # for example a rate-limited planner call
        return {"ranked_sources": [], "top_score": None, "answer": None, "failed": True,
                "search_term": None, "intent": None, "error": str(exc)[:200]}

    if state["current_query"] == "CONVERSATIONAL":  # the planner decided not to retrieve
        return {"ranked_sources": [], "top_score": None, "answer": None, "failed": False,
                "search_term": None, "intent": "conversational"}

    query_used = choose_search_query(state, settings.QUERY_REWRITE, settings.CLEAN_REWRITE)
    state.update(retrieve_node(state))
    citations = state.get("citations") or []
    scores = [c["score"] for c in citations if c.get("score") is not None]
    return {
        "ranked_sources": [c["source"] for c in citations],
        "top_score": max(scores) if scores else None,
        "answer": None,
        "failed": False,
        "search_term": state["current_query"],   # the planner's rewrite
        "query_used": query_used,                # what the retriever actually searched with
        "intent": "technical",
    }


def _post_with_retry(url: str, question: str, index: int) -> dict:
    """POST to /query, retrying transient failures. Returns the last response if all attempts fail."""
    import requests

    last = {"status": "error", "answer": ""}
    for delay in (0, *RETRY_DELAYS):
        time.sleep(delay)
        try:
            resp = requests.post(
                f"{url.rstrip('/')}/query",
                json={"q": question, "thread_id": f"retrieval_eval_{index}"},
                timeout=API_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            last = {"status": "error", "answer": str(exc)}
            continue
        if data.get("status") != "error":  # the API reports failures as HTTP 200 + status "error"
            return data
        last = data
    return last


def _search_term(data: dict) -> str | None:
    """The planner's rewritten query, taken from the thought process."""
    for step in data.get("thought_process") or []:
        if isinstance(step, str) and step.startswith("Search Term:"):
            return step.split(":", 1)[1].strip()
    return None


def run_api(question: str, index: int, url: str) -> dict:
    data = _post_with_retry(url, question, index)
    citations = data.get("citations") or []
    scores = [c["score"] for c in citations if c.get("score") is not None]
    return {
        "ranked_sources": [c["source"] for c in citations],
        "top_score": max(scores) if scores else None,
        "answer": data.get("answer") or "",
        "failed": data.get("status") == "error",
        "search_term": _search_term(data),
    }


def _config(args) -> dict:
    config = {"mode": args.mode, "k": args.k, "candidates": CANDIDATES, "rerank": not args.no_rerank}
    if args.mode == "agent":
        config["rerank"] = True  # the agent always reranks, and returns its top 5
        config["rewrite"] = args.rewrite
        config["clean_rewrite"] = not getattr(args, "no_clean_rewrite", False)
    if args.mode in ("retriever", "agent"):
        from app.config import settings

        config["embedding_model"] = settings.EMBEDDING_MODEL
        config["collection"] = settings.QDRANT_COLLECTION
    return config


def _report(args, questions, by_id, config, stopped_early) -> dict:
    ordered = [by_id[q["id"]] for q in questions if q["id"] in by_id]
    summary = summarize(ordered, args.k)
    summary["n_missing"] = len(questions) - len(ordered)
    return {
        "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": summary["n_missing"] == 0 and summary["n_failed"] == 0,
        "stopped_early": stopped_early,
        "config": config,
        "summary": summary,
        "results": ordered,
    }


def run(args, out_path: Path | None = None, previous: list[dict] | None = None) -> dict:
    """
    Score every question. `previous` holds results from an earlier run: those that
    did not fail are reused. When `out_path` is given the report is saved after
    every question.
    """
    questions = load_questions()
    config = _config(args)
    if args.mode == "agent":
        from app.config import settings

        settings.QUERY_REWRITE = args.rewrite  # read by the retriever node at call time
        settings.CLEAN_REWRITE = not getattr(args, "no_clean_rewrite", False)
    by_id = {r["id"]: r for r in (previous or []) if not r.get("failed")}
    consecutive_failures = 0
    stopped_early = False

    for i, q in enumerate(questions):
        if q["id"] in by_id:
            continue
        if args.mode == "retriever":
            outcome = run_retriever(q["question"], args.k, rerank=not args.no_rerank)
        elif args.mode == "agent":
            outcome = run_agent(q["question"])
            time.sleep(AGENT_DELAY_SECONDS)
        else:
            outcome = run_api(q["question"], i, args.url)
            time.sleep(API_DELAY_SECONDS)
        by_id[q["id"]] = {"id": q["id"], "gold": gold_sources(q), **outcome}
        consecutive_failures = consecutive_failures + 1 if outcome.get("failed") else 0
        print(f"[{i + 1}/{len(questions)}] {q['id']}{'  FAILED' if outcome.get('failed') else ''}", flush=True)

        if out_path:
            out_path.write_text(json.dumps(_report(args, questions, by_id, config, False), indent=2) + "\n")
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            stopped_early = True
            print(f"\nStopping: {consecutive_failures} questions failed in a row. "
                  "This usually means a rate limit or daily quota. Re-run with --resume later.")
            break

    return _report(args, questions, by_id, config, stopped_early)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="name for this run, used as the output file name")
    parser.add_argument("--mode", choices=["retriever", "agent", "api"], default="retriever")
    parser.add_argument("--rewrite", choices=["always", "followups"], default="followups",
                        help="agent mode: always search with the planner's rewrite (old) or only on follow-ups (new)")
    parser.add_argument("--k", type=int, default=5, help="rank cut-off for the metrics")
    parser.add_argument("--no-clean-rewrite", action="store_true",
                        help="agent mode: do not repair defects in the planner's rewrite (old behaviour)")
    parser.add_argument("--no-rerank", action="store_true", help="retriever mode: skip FlashRank")
    parser.add_argument("--url", default="http://localhost:8000", help="api mode: backend base URL")
    parser.add_argument("--resume", action="store_true", help="reuse non-failed results from an earlier run with this label")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    previous = None
    if args.resume and out.exists():
        previous = json.loads(out.read_text())["results"]
        print(f"Resuming: {sum(1 for r in previous if not r.get('failed'))} good results kept.")

    report = run(args, out_path=out, previous=previous)
    out.write_text(json.dumps(report, indent=2) + "\n")

    s = report["summary"]
    print(f"\nwrote {out.relative_to(EVALS_DIR.parent)}  (complete: {report['complete']})")
    print(f"  scored {s['n_answerable']} answerable + {s['n_unanswerable']} unanswerable | "
          f"failed {s['n_failed']} | not yet run {s['n_missing']}")
    if s["n_answerable"]:
        for name in ("hit", "recall", "precision", "mrr"):
            lo, hi = s[name]["ci95"]
            print(f"  {name}@{s['k']:<2} {s[name]['mean']:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
    if "refusal_rate_on_unanswerable" in s:
        r = s["refusal_rate_on_unanswerable"]
        print(f"  refusal rate on unanswerable: {r['mean']:.2f} (n={r['n']})")
    if "suggested_score_threshold" in s:
        t = s["suggested_score_threshold"]
        print(f"  suggested score threshold: {t['threshold']:.3f} (balanced accuracy {t['balanced_accuracy']:.2f})")
    if not report["complete"]:
        print("  NOTE: this run is incomplete, so the numbers above cover only the scored questions.")


if __name__ == "__main__":
    main()
