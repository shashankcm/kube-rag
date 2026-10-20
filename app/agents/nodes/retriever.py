import logfire

from app.agents.state import AgentState
from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_hits


def choose_search_query(state: AgentState, mode: str) -> str:
    """
    Picks the text to search with.

    The planner rewrites the question into a search query. On a first question that rewrite
    pads it with generic words ("Kubernetes ... documentation") and measurably lowers ranking
    quality, so the user's own words are used instead. With earlier conversation the rewrite is
    kept, because it resolves references like "what about its limits?".
    """
    rewritten = state["current_query"]
    messages = state.get("messages") or []
    if mode == "always" or not messages:
        return rewritten
    if len(messages) == 1:  # a first question: no earlier turns to resolve references against
        return messages[0]["content"]
    return rewritten


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.

    Returns the chunk texts as `documents` and their metadata as `citations`.
    """
    query = choose_search_query(state, settings.QUERY_REWRITE)

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
