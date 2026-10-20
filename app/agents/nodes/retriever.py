import re

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_hits


def clean_search_query(text: str) -> str:
    """
    Repairs defects seen in real planner output (29 of 75 rewrites in evals/results/agent-always.json):
    stray control characters such as a trailing NUL, the whole query repeated twice with no
    separator, and web-search operators (site:..., OR) that a vector index cannot use.
    """
    text = "".join(c for c in text if c.isprintable() or c.isspace())  # drops \x00 and friends
    text = " ".join(text.split())
    half = len(text) // 2
    # Only collapse when the WHOLE string is one half twice, so ordinary repeated words survive.
    if len(text) >= 20 and len(text) % 2 == 0 and text[:half] == text[half:]:
        text = text[:half]
    text = re.sub(r"\bsite:\S+", "", text)
    return " ".join(t for t in text.split() if t not in ("OR", "AND"))


def choose_search_query(state: AgentState, mode: str, clean: bool = True) -> str:
    """
    Picks the text to search with.

    The planner rewrites the question into a search query. On a first question that rewrite
    pads it with generic words ("Kubernetes ... documentation") and measurably lowers ranking
    quality, so the user's own words are used instead. With earlier conversation the rewrite is
    kept, because it resolves references like "what about its limits?". The rewrite is cleaned
    first (see clean_search_query); if nothing usable is left the user's own words are used.
    """
    rewritten = state["current_query"]
    messages = state.get("messages") or []
    users_words = messages[-1]["content"] if messages else ""

    def rewrite() -> str:
        text = clean_search_query(rewritten) if clean else rewritten
        return text or users_words or rewritten

    if mode == "always" or not messages:
        return rewrite()
    if len(messages) == 1:  # a first question: no earlier turns to resolve references against
        return users_words
    return rewrite()


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.

    Returns the chunk texts as `documents` and their metadata as `citations`.
    """
    query = choose_search_query(state, settings.QUERY_REWRITE, settings.CLEAN_REWRITE)

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        if not raw_results:
            logfire.warning("No candidates retrieved; the responder will not guess.")
            return {
                "documents": [],
                "citations": [],
                "status": "No relevant context found.",
                "plan": state["plan"] + ["No Context Found"],
            }

        with logfire.span("⚖️ Semantic Reranking"):
            top_hits = rerank_hits(query, raw_results, top_n=5)
            logfire.info(f"Reranking complete. Kept top {len(top_hits)} most relevant chunks.")

    return {
        "documents": [hit["content"] for hit in top_hits],
        "citations": [
            {
                "source": hit.get("source", "Unknown"),
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("rerank_score", hit.get("score")),
            }
            for hit in top_hits
        ],
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
