import time

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2
_RATE_LIMIT_ERRORS = ["429", "rate", "quota", "resource_exhausted"]

_active_model = None
_model_type: str | None = None


def _probe_gemini() -> GoogleGenerativeAIEmbeddings | None:
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim)")
        return model
    except Exception as e:
        logfire.error(
            f"Gemini probe failed: {e}. Will use sentence-transformers fallback."
        )
        return None


def _load_fallback() -> SentenceTransformer:
    """Load the sentence-transformers fallback model."""
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    global _active_model, _model_type

    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


def get_embedding_dim() -> int:
    """Return the embedding dimension for the active model. Call after init."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a batch of text using the active model."""
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(
                    rate_limit in err for rate_limit in _RATE_LIMIT_ERRORS
                )
                if is_rate_limit and attempt < 3:
                    wait = 2**attempt
                    logfire.warning(
                        f"Gemini rate limit hit - retrying in {wait} seconds"
                        f" (attempt {attempt + 1}/4)"
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError(f"Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query using the active model."""
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    else:
        return _active_model.encode([query], show_progress_bar=False).tolist()


def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed a single text using the active model."""
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        with logfire.span(f"Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
