"""Historical news for a ticker over a date window - the single news-retrieval layer.

This is the ONE function everything calls for company news (both the forecast-vetting path and the
"why did it move" explanation tool). It reaches back months so a backtest can ask "what was the news
just before this event?", and it routes by market:

  US:  Finnhub company-news - ticker-native (symbol + from/to), free tier ~60 req/min and about a
       year of history, covering our Jan-2026-onward scope. Needs FINNHUB_API_KEY.
       Fallback: GDELT (core/gdelt.py) - keyless but rate-limited and full-text (noisier), used when
       no Finnhub key is set so the pipeline still runs before the key is added.
  PH:  Google News RSS by company name (via market_data), since Finnhub/GDELT don't cover the PSE.
       RSS isn't date-queryable, so PH is best-effort: recent items are filtered to the window.

All sources are normalised to the SAME shape {title, url, domain, seendate, ts, source} and this
module owns the on-disk cache (immutable past windows). Retrieval is neutral about lookahead: any
no-lookahead cut (e.g. the market-open cut for vetting) is applied by the CALLER, not here, so the
explanation tool can read same-day/intraday news while the vetting tool cuts at the open.
"""
import datetime as dt
import json
import threading
import time
from email.utils import parsedate_to_datetime

import requests

from . import config, gdelt

_cache_lock = threading.Lock()
_cache = None

# light spacing so we stay well under Finnhub's 60/min even on an uncached backtest sweep
_MIN_INTERVAL = 1.1
_last_req = [0.0]
_req_lock = threading.Lock()


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(config.NEWS_CACHE_PATH) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _throttle():
    with _req_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.monotonic()


def _iso(x) -> str:
    """A date-ish value -> 'YYYY-MM-DD'."""
    if isinstance(x, (dt.date, dt.datetime)):
        return x.strftime("%Y-%m-%d")
    s = str(x)
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _finnhub_news(ticker: str, start: str, end: str, max_records: int) -> list[dict]:
    """Finnhub company-news for [start, end]. Returns [] on any failure so the caller can fall back."""
    if not config.FINNHUB_API_KEY:
        return []
    params = {"symbol": ticker.upper().strip(), "from": _iso(start), "to": _iso(end),
              "token": config.FINNHUB_API_KEY}
    _throttle()
    try:
        r = requests.get(f"{config.FINNHUB_BASE_URL}/company-news", params=params, timeout=20)
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, list):   # errors come back as a dict, e.g. {"error": "..."}
        return []
    # Finnhub tags articles to a symbol loosely: ~60% of "company-news?symbol=X" headlines don't
    # actually mention X (generic market-wrap, other companies). Filter to headlines that name the
    # company, same relevance check the GDELT path uses, so we don't feed the agent off-topic noise.
    toks = gdelt._COMPANY.get(ticker.upper().strip(), {}).get("tokens")
    items = []
    for a in data[:max_records]:
        headline = a.get("headline")
        ts = a.get("datetime")
        if not headline or not ts:
            continue
        if toks and not any(t in headline.lower() for t in toks):
            continue
        # keep the precise unix timestamp (UTC), not just the date - the overnight/pre-open window
        # in vet_forecast needs sub-day resolution to separate pre-move news from post-move news.
        seen = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")
        items.append({"title": headline, "url": a.get("url"),
                      "domain": a.get("source"), "seendate": seen, "ts": int(ts)})
    return items


def _ph_news_ranged(ticker: str, s_iso: str, e_iso: str, max_records: int) -> list[dict]:
    """PH company news via Google News RSS (market_data._ph_news), normalised to the canonical shape
    and client-side filtered to [s_iso, e_iso]. RSS is not date-queryable, so this is best-effort:
    it returns whichever recent items fall inside the window."""
    from . import market_data, ph_data   # local import keeps module load order simple
    name = ph_data.KNOWN_PSE.get(ticker.upper().strip(), ticker)
    out = []
    for n in market_data._ph_news(name, max_records):
        d = None
        if n.get("published"):
            try:
                d = parsedate_to_datetime(str(n["published"]))
            except (TypeError, ValueError):
                d = None
        seen = d.strftime("%Y-%m-%d") if d else None
        if seen and not (s_iso <= seen <= e_iso):
            continue
        out.append({"title": n.get("title"), "url": n.get("link"), "domain": n.get("publisher"),
                    "seendate": seen, "ts": (int(d.timestamp()) if d else None),
                    "source": "googlenews"})
    return out


def news_for_ticker(ticker: str, start, end, max_records: int = 50) -> list[dict]:
    """Company headlines for a ticker over [start, end] as [{title,url,domain,seendate,ts,source}].
    Routes US->Finnhub(then GDELT), PH->Google News RSS; result cached per immutable window (US)."""
    ticker = ticker.upper().strip()
    s_iso, e_iso = _iso(start), _iso(end)

    from . import market_data   # local import avoids any import cycle at module load
    if market_data.market_of(ticker) == "PH":
        return _ph_news_ranged(ticker, s_iso, e_iso, max_records)

    key = f"{ticker}:{s_iso}:{e_iso}"
    today = dt.date.today().isoformat()
    cache = _load_cache()
    if key in cache and e_iso < today:   # only trust cache for a fully-past window
        return cache[key]

    items = _finnhub_news(ticker, s_iso, e_iso, max_records)
    source = "finnhub"
    if not items:   # no key, empty, or error -> keyless fallback
        items = gdelt.news_for_ticker(ticker, s_iso, e_iso, max_records)
        source = "gdelt"

    # enforce the date window ourselves (a finer timestamp cut happens in vet_forecast)
    items = [n for n in items if n.get("seendate") and s_iso <= n["seendate"] <= e_iso]
    for n in items:
        n["source"] = source
        n.setdefault("ts", None)   # gdelt fallback has date-only granularity

    if e_iso < today:
        cache[key] = items
        with _cache_lock:
            try:
                with open(config.NEWS_CACHE_PATH, "w") as f:
                    json.dump(cache, f)
            except Exception:
                pass
    return items
