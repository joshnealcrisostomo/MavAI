"""All runtime config comes from env vars so the same image works locally and in the cloud."""
import os
from pathlib import Path


def _load_dotenv():
    """Load stock-agent/.env into the environment if present (no python-dotenv dependency).

    Uses setdefault so real env vars (e.g. those docker-compose passes in) always win. This is
    what lets `streamlit run` / `uvicorn` pick up the Groq key locally without extra setup.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

# Groq exposes an OpenAI-compatible API, so we keep using the OpenAI SDK and just
# repoint base_url + key + model. Swap these envs to use OpenAI/Gemini instead.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")


def _parse_keys():
    """One or more API keys. Accepts a comma-separated list so we can rotate across several
    free Groq keys and fail over when one hits its rate limit. Any of these env vars works:
    GROQ_API_KEYS (preferred for a list), GROQ_API_KEY, or LLM_API_KEY - each may hold a
    comma-separated list too."""
    raw = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


LLM_API_KEYS = _parse_keys()
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else ""  # first key (kept for backward compat)

# two-model split: the big one reasons and calls tools, the small one does the cheap
# classification jobs (disambiguation, guardrail topic check, memory summaries)
AGENT_MODEL = os.getenv("AGENT_MODEL", "llama-3.3-70b-versatile")
# NOTE: Groq decommissions llama-3.1-8b-instant on 2026-08-16; replacement is openai/gpt-oss-20b.
# Kept on the 8B through the 2026-08-15 demo (known-good, tested); swap the default after.
FAST_MODEL = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")

# where chroma keeps the reference store (ticker info + finance glossary) inside the container
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

# mlflow writes runs here (file store keeps the image light - no server needed)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "stock-agent")

# USD per 1M tokens, used to estimate cost_usd per request for the LLMOps log.
# Numbers are Groq's published rates; tweak if you switch models/providers.
PRICING = {
    "openai/gpt-oss-20b":      {"input": 0.10, "output": 0.50},   # verify Groq's current rate
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},   # decommissioned 2026-08-16 (kept for old logs)
}

# Philippine history (phisix) fetch settings. PH has no bulk endpoint, so a range is fetched
# one trading day at a time - we cap the range, fetch in parallel, and cache immutable past days.
PH_CACHE_PATH = os.getenv("PH_CACHE_PATH", "./ph_cache.json")
PH_MAX_RANGE_DAYS = int(os.getenv("PH_MAX_RANGE_DAYS", "400"))
PH_FETCH_WORKERS = int(os.getenv("PH_FETCH_WORKERS", "10"))

# how many recent turns the memory keeps verbatim before summarising
BUFFER_WINDOW = int(os.getenv("BUFFER_WINDOW", "4"))
# how many reference chunks RAG returns per lookup
RAG_K = int(os.getenv("RAG_K", "3"))
# safety cap on the ReAct loop so a confused model can't spin forever
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "6"))

# Forecast-vetting (Component #14). Which forecast backend the LIVE vet_forecast tool uses.
# The bake-off (experiments/forecast_bakeoff.py) picks the winner. On FAANG 2026, Theta was the
# best REAL model on directional accuracy (~47%), and notably NO model beat the ~50% naive
# coin-flip baseline - even Chronos (the foundation model). We ship Theta (a real statistical DS
# model, torch-free); Chronos stays an experiment-only backend. One of: naive, ets, arima, theta,
# chronos.
FORECAST_BACKEND = os.getenv("FORECAST_BACKEND", "theta")
# default horizon (trading days ahead) a forecast/vetting looks at
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "5"))
# vetting sensitivity, named by who it serves. "investor" (only flag genuine threats - precise, few
# false alarms; for long-term holders + automation; the default) or "analyst" (flag ANY material
# catalyst, incl. good news - wide net to triage; for active due-diligence + risk desks). This is a
# user-facing setting (a UI toggle), not something the LLM picks mid-conversation.
VET_MODE = os.getenv("VET_MODE", "investor")
# which pretrained Chronos-Bolt size to load if the chronos backend is used (small is a good
# CPU default; mini is lighter, base is more accurate)
CHRONOS_MODEL = os.getenv("CHRONOS_MODEL", "amazon/chronos-bolt-small")

# Historical news for the "why/vet" reasoning. Primary source is Finnhub (ticker-native
# company-news, free tier ~60 req/min and ~1yr history). GDELT is the keyless fallback.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1")
NEWS_CACHE_PATH = os.getenv("NEWS_CACHE_PATH", "./news_cache.json")

# GDELT fallback (keyless). ArtList mode only searches the last 3 months of a window, so we always
# query narrow per-event windows - that also enforces "news strictly before the event".
GDELT_BASE_URL = os.getenv("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
GDELT_CACHE_PATH = os.getenv("GDELT_CACHE_PATH", "./gdelt_cache.json")
