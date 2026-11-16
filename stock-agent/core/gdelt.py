"""Historical news via the GDELT DOC 2.0 API.

The live news feed in market_data (yfinance / Google News RSS) only returns *recent* headlines,
which is useless for a backtest that needs to know what the news was months ago. GDELT is a free,
open archive of worldwide news going back years, so we use it to fetch the headlines that existed
in a narrow window BEFORE a chosen event date.

Two things matter about the DOC 2.0 ArtList endpoint:
  1. It only searches the *last 3 months* of whatever window you give it. We always pass a narrow
     per-event window (e.g. event-14d .. event-1d), so that limitation never bites - and a short
     window ending the day before the event also enforces "news strictly before the move".
  2. It returns article METADATA (title, url, domain, seendate), not full text. That's fine - the
     agent reasons over headlines, same as the existing get_price_move_context path.

Results are cached to disk (immutable past windows) exactly like market_data's ph_cache.
"""
import datetime as dt
import json
import threading
import time

import requests

from . import config

# GDELT asks for no more than one request every 5 seconds (it returns HTTP 429 otherwise). We
# space calls out ourselves and retry on a 429. Immutable windows are cached to disk, so this
# throttle is only paid once per unique (ticker, window) - re-runs of the backtest are instant.
_MIN_INTERVAL = 6.0
_last_req = [0.0]
_req_lock = threading.Lock()


def _throttle():
    with _req_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.monotonic()

# Per-ticker search query + the title tokens we require client-side to cut off-topic hits
# (e.g. "Apple" the fruit, "Amazon" the river). Query uses the legal/common name OR the ticker.
_COMPANY = {
    "AAPL": {"query": '("Apple Inc" OR "Apple" OR AAPL)', "tokens": ["apple", "aapl"]},
    "MSFT": {"query": '("Microsoft" OR MSFT)', "tokens": ["microsoft", "msft"]},
    "GOOG": {"query": '("Alphabet" OR "Google" OR GOOG OR GOOGL)', "tokens": ["alphabet", "google", "goog"]},
    "GOOGL": {"query": '("Alphabet" OR "Google" OR GOOG OR GOOGL)', "tokens": ["alphabet", "google", "goog"]},
    "AMZN": {"query": '("Amazon.com" OR "Amazon" OR AMZN)', "tokens": ["amazon", "amzn"]},
    "META": {"query": '("Meta Platforms" OR "Facebook" OR "Instagram" OR META)', "tokens": ["meta", "facebook", "instagram", "zuckerberg"]},
    "NVDA": {"query": '("Nvidia" OR NVDA)', "tokens": ["nvidia", "nvda"]},
    "TSLA": {"query": '("Tesla Inc" OR "Tesla" OR TSLA)', "tokens": ["tesla", "tsla", "musk"]},
}

_cache_lock = threading.Lock()
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(config.GDELT_CACHE_PATH) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _to_gdelt_dt(x, end=False) -> str:
    """Normalise a date to GDELT's YYYYMMDDHHMMSS. Accepts 'YYYY-MM-DD', 'YYYYMMDD',
    a full 'YYYYMMDDHHMMSS', or a date/datetime. A bare date maps to 00:00:00 (start) or
    23:59:59 (end)."""
    if isinstance(x, (dt.date, dt.datetime)):
        x = x.strftime("%Y%m%d")
    s = str(x).replace("-", "").replace(":", "").replace("T", "").replace(" ", "").rstrip("Z")
    if len(s) == 14:
        return s
    if len(s) == 8:
        return s + ("235959" if end else "000000")
    raise ValueError(f"unrecognised date for GDELT: {x}")


def _seendate_to_iso(seendate: str) -> str:
    """GDELT seendate looks like '20260424T130000Z' -> '2026-04-24'."""
    d = str(seendate)[:8]
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else None


def fetch_news(query: str, start, end, max_records: int = 50) -> list[dict]:
    """Low-level GDELT DOC 2.0 ArtList call. `start`/`end` may be ISO dates or GDELT datetimes.
    Any failure (network, non-JSON error page) returns [] - callers treat that as 'no grounding'."""
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(min(int(max_records), 250)),
        "startdatetime": _to_gdelt_dt(start),
        "enddatetime": _to_gdelt_dt(end, end=True),
        "sort": "datedesc",
    }
    data = None
    for attempt in range(3):
        _throttle()
        try:
            r = requests.get(config.GDELT_BASE_URL, params=params, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429 or r.text.lstrip().startswith("Please limit"):
                time.sleep(_MIN_INTERVAL)   # tripped the rate limit - back off and retry
                continue
            data = r.json()
            break
        except Exception:
            return []
    if data is None:
        return []
    items = []
    for a in data.get("articles", []):
        title = a.get("title")
        if not title:
            continue
        items.append({
            "title": title,
            "url": a.get("url"),
            "domain": a.get("domain"),
            "seendate": _seendate_to_iso(a.get("seendate")),
        })
    return items


def news_for_ticker(ticker: str, start, end, max_records: int = 50) -> list[dict]:
    """Pre-event headlines for a FAANG ticker over [start, end]. Builds the search query from the
    per-ticker map, filters titles to those actually mentioning the company, and caches immutable
    past windows to disk."""
    ticker = ticker.upper().strip()
    spec = _COMPANY.get(ticker)
    if spec is None:
        return []

    s_iso = _to_gdelt_dt(start)[:8]
    e_iso = _to_gdelt_dt(end, end=True)[:8]
    key = f"{ticker}:{s_iso}:{e_iso}"
    today = dt.date.today().strftime("%Y%m%d")
    cache = _load_cache()
    if key in cache and e_iso < today:   # only trust cache for a fully-past window
        return cache[key]

    raw = fetch_news(spec["query"], start, end, max_records)
    toks = spec["tokens"]

    def _in_window(n):
        # GDELT can return a straggler just past the window edge; enforce the window ourselves so
        # a cited headline can never postdate the event we're vetting.
        sd = (n.get("seendate") or "").replace("-", "")
        return bool(sd) and s_iso <= sd <= e_iso

    filtered = [n for n in raw
                if n["title"] and _in_window(n) and any(t in n["title"].lower() for t in toks)]

    if e_iso < today:   # cache only immutable windows
        cache[key] = filtered
        with _cache_lock:
            try:
                with open(config.GDELT_CACHE_PATH, "w") as f:
                    json.dump(cache, f)
            except Exception:
                pass
    return filtered
