"""Known stock entities (US + PH) for deterministic routing.

The small 8B classifier used by the guardrail and disambiguation steps doesn't reliably know
that "Globe" or "Jollibee" are stocks (weak on Philippine names). So before falling back to the
model, both gates check here: if the message clearly names a company or ticker we know, treat it
as a valid stock query and proceed. This keeps named companies from being wrongly blocked or
sent back for clarification.
"""
import re

from . import ph_data

# US tickers we seed / commonly get asked about
_US_TICKERS = {"AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V",
               "WMT", "DIS", "KO", "INTC", "AMD", "NFLX"}

# lowercase name keywords that signal a known company (US + PH)
_NAME_KEYWORDS = {
    # US
    "apple", "microsoft", "google", "alphabet", "amazon", "nvidia", "tesla", "meta",
    "facebook", "instagram", "netflix", "intel", "visa", "walmart", "disney", "coca-cola",
    "coca cola", "jpmorgan", "jp morgan",
    # PH
    "globe", "pldt", "bdo", "unibank", "bpi", "metrobank", "ayala", "jollibee", "meralco",
    "universal robina", "aboitiz", "jg summit", "puregold", "dmci", "gt capital", "sm investments",
    "sm prime", "san miguel",
}

# all known tickers, upper-case, matched with word boundaries
_ALL_TICKERS = _US_TICKERS | set(ph_data.KNOWN_PSE.keys())
_TICKER_RE = re.compile(r"\b(" + "|".join(sorted(_ALL_TICKERS, key=len, reverse=True)) + r")\b")
_CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,5}\b")  # $AAPL style


def mentions_known_stock(text: str) -> bool:
    """True if the text names a company or ticker we recognise (US or PH)."""
    if _CASHTAG_RE.search(text):
        return True
    if _TICKER_RE.search(text):          # tickers are case-sensitive (avoid matching words like "V")
        return True
    low = text.lower()
    return any(kw in low for kw in _NAME_KEYWORDS)
