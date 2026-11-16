"""RAG module: a small reference store the agent grounds concept answers in.

Two kinds of seed knowledge:
  - ticker -> company / sector facts (helps both definitions and disambiguation)
  - a finance-term glossary (moving average, OHLC, P/E, market cap, volatility, ...)

Embeddings use Chroma's DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX runtime) so we
avoid pulling in PyTorch and the Docker image stays small

Price questions are answered with live tools; this store is for "what is / what does X do"
questions and as a hint source for name->ticker grounding.
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from . import config, ph_data

_client = chromadb.PersistentClient(
    path=config.CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False),  # silence posthog telemetry noise
)
_ef = embedding_functions.DefaultEmbeddingFunction()  # ONNX MiniLM, no torch
_kb = _client.get_or_create_collection("stock_reference", embedding_function=_ef)

# ticker -> company / sector reference for common US names
_TICKERS = [
    "AAPL is Apple Inc., a technology company in consumer electronics (iPhone, Mac, services).",
    "MSFT is Microsoft, a technology company in software, cloud (Azure), and productivity tools.",
    "GOOG and GOOGL are Alphabet, Google's parent, in online advertising, search, and cloud.",
    "AMZN is Amazon, in e-commerce and cloud computing (AWS).",
    "NVDA is NVIDIA, a semiconductor company making GPUs for gaming and AI data centers.",
    "TSLA is Tesla, an electric-vehicle and clean-energy company.",
    "META is Meta Platforms, parent of Facebook, Instagram, and WhatsApp, in social media and ads.",
    "JPM is JPMorgan Chase, a large US bank in financial services.",
    "V is Visa, a payments-technology company running a card network.",
    "WMT is Walmart, a retail company running large discount stores.",
    "DIS is The Walt Disney Company, in media, streaming, and theme parks.",
    "KO is The Coca-Cola Company, a beverage company.",
    "INTC is Intel, a semiconductor company making CPUs.",
    "AMD is Advanced Micro Devices, a semiconductor company (CPUs and GPUs).",
    "NFLX is Netflix, a streaming-entertainment company.",
]

# ticker -> company reference for Philippine Stock Exchange (PSE) names. Tags them as PH so the
# agent and the guardrail/disambiguation gates recognise them; the tools then fetch PH data
# automatically through the unified market_data layer (no separate PH tool needed).
#
# Built from ph_data.KNOWN_PSE (itself fetched live from phisix's own /stocks.json roster,
# with a disk cache and hardcoded fallback - see ph_data.py) instead of a hand-picked list.
# This is what lets a query like "did vivant go up this week" resolve correctly: "WT is
# Vivant Corporation" is embedded and retrievable even though nobody typed "Vivant" into this
# file by hand. Coverage grows/shrinks automatically as phisix's tracked roster changes.
_PH_TICKERS = [
    f"{ticker} is {name}, a Philippine (PSE) company."
    for ticker, name in sorted(ph_data.KNOWN_PSE.items())
]

# finance-term glossary
_GLOSSARY = [
    "A moving average is the average closing price over a rolling window of days (e.g. a 5-day "
    "or 20-day MA), used to smooth out short-term noise and show the trend.",
    "OHLC stands for Open, High, Low, Close - the four key prices of a trading day, shown on a "
    "candlestick chart.",
    "Market capitalization (market cap) is a company's share price times its number of shares "
    "outstanding; it measures the company's total market value.",
    "The P/E ratio (price-to-earnings) is the share price divided by earnings per share; it "
    "gauges how expensive a stock is relative to its profits.",
    "Volume is the number of shares traded in a period; higher volume means more activity.",
    "Volatility measures how much a price swings over time; higher volatility means bigger swings.",
    "A 52-week high/low is the highest/lowest price a stock traded at over the past year.",
    "A dividend is a cash payment some companies make to shareholders out of profits.",
    "A bull market is a period of rising prices; a bear market is a period of falling prices.",
    "Percentage change is (last price minus first price) divided by first price, times 100.",
]

_SEED = _TICKERS + _PH_TICKERS + _GLOSSARY


def _seed():
    # (re)seed when empty OR when the seed set changed size (e.g. we added the PH names).
    # a persisted store from an earlier run would otherwise keep the old, smaller seed.
    if _kb.count() != len(_SEED):
        if _kb.count() > 0:
            _kb.delete(_kb.get()["ids"])
        # takes each chunk in _SEED, runs it through _ef (MiniLM) to get its embedding vector
        # stores both the text and its vector inside the collection
        _kb.add(documents=_SEED, ids=[f"seed_{i}" for i in range(len(_SEED))])


_seed()

# Retrieval
def retrieve(query: str, k: int = None) -> str:
    """Semantic top-k search; returns a bullet list of matching reference chunks."""
    k = k or config.RAG_K
    if _kb.count() == 0:
        return ""
    res = _kb.query(query_texts=[query], n_results=min(k, _kb.count()))
    return "\n".join(f"- {d}" for d in res["documents"][0])


def doc_count() -> int:
    return _kb.count()