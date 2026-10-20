import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    # --- EMBEDDINGS ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_kub_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("PORTKEY_GROQ_SLUG", "kube-rage-groq")  # Portkey Groq integration slug
    GROQ_SLUG_2 = os.getenv("PORTKEY_GROQ_SLUG_2", GROQ_SLUG)  # fallback integration (same by default)
    # Optional saved Portkey config ("pc-...") — enables fallback/cache/retry when inline configs are disabled
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

    # --- RETRIEVAL ---
    # "followups": search with the user's own words on a first question and use the planner's
    # rewrite only when there is earlier conversation (where it resolves "what about its limits?").
    # "always": always search with the planner's rewrite (the original behaviour).
    QUERY_REWRITE = os.getenv("QUERY_REWRITE", "followups")
    # Repair known defects in the planner's rewrite (control characters, repeated text, site: operators).
    # Turn off only to reproduce the old behaviour for an experiment.
    CLEAN_REWRITE = os.getenv("CLEAN_REWRITE", "true").lower() != "false"

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "kube-rag")
    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )


# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "kube-rag")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
)

settings = Settings()
