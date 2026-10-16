import time

import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker


def rerank_hits(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-scores retrieval hits against the query and returns the best `top_n`.

    Each hit is a dict with at least `content`; metadata such as `source` and
    `chunk_id` is carried through, and `rerank_score` is added.
    """
    if not hits:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(hits)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()

        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [{"id": i, "text": hit["content"]} for i, hit in enumerate(hits)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))

        # Results come back sorted by highest score first; 'id' maps to the input hit
        reranked = [
            {**hits[res["id"]], "rerank_score": float(res["score"])}
            for res in results[:top_n]
        ]

        duration = time.time() - start_time
        top_score = reranked[0]["rerank_score"] if reranked else "N/A"
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        return reranked

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fall back to the original Qdrant order so the user still gets an answer
        return hits[:top_n]


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """String-only wrapper around `rerank_hits`, kept for existing callers."""
    hits = [{"content": doc} for doc in documents]
    return [hit["content"] for hit in rerank_hits(query, hits, top_n)]
