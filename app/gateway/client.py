import os

import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


PRIMARY_MODEL = f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"
FALLBACK_MODEL = f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"

# Gateway config (used only as a fallback when PORTKEY_CONFIG_ID is unset; the workspace may reject inline configs):
#   - Fallback: primary gpt-oss-120b → gpt-oss-20b on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": PRIMARY_MODEL}},
        {"override_params": {"model": FALLBACK_MODEL}},
    ]
}

# Saved config ID if provided; otherwise no config (direct calls, no fallback/cache).
# The inline GATEWAY_CONFIG is kept for workspaces that allow it: set PORTKEY_USE_INLINE_CONFIG=1.
if settings.PORTKEY_CONFIG_ID:
    ACTIVE_CONFIG = settings.PORTKEY_CONFIG_ID
elif os.getenv("PORTKEY_USE_INLINE_CONFIG") == "1":
    ACTIVE_CONFIG = GATEWAY_CONFIG
else:
    ACTIVE_CONFIG = None

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=ACTIVE_CONFIG
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PRIMARY_MODEL,
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=ACTIVE_CONFIG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"