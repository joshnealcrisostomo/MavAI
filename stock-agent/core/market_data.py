"""Unified market-data layer: one interface over yfinance (US) and phisix (PH).

The analytics in tools.py (min/max/average, moving average, comparison) only need a series of
daily closes. Both markets can produce that, so we normalise them here and the tool logic runs
once for either market:

  daily_series(ticker, start, end) -> {ok, ticker, market, currency, has_ohlc, series:[...]}

US (yfinance) carries full OHLC per day, so US stocks can be drawn as candlesticks. PH (phisix)
only exposes a daily close + volume, and has no bulk endpoint - a range is fetched one trading
day at a time, in parallel, with immutable past days cached to disk so repeat queries are instant.
"""
import datetime as dt
import json
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
import yfinance as yf

from . import config, ph_data

_US_MAX_RANGE_DAYS = 366 * 5


def market_of(ticker: str) -> str:
    """Return 'PH' for a known Philippine ticker, else 'US'."""
    return "PH" if ticker.upper().strip() in ph_data.KNOWN_PSE else "US"


# US path (yfinance) - full OHLC
def _us_series(ticker, s, e):
    df = yf.Ticker(ticker).history(start=s.isoformat(),
                                   end=(e + dt.timedelta(days=1)).isoformat(),
                                   auto_adjust=True)
    if df.empty:
        return None
    series = [{
        "date": idx.date().isoformat(),
        "open": round(float(r["Open"]), 2), "high": round(float(r["High"]), 2),
        "low": round(float(r["Low"]), 2), "close": round(float(r["Close"]), 2),
        "volume": int(r["Volume"]),
    } for idx, r in df.iterrows()]
    return {"currency": "USD", "has_ohlc": True, "series": series}


# PH path (phisix) - close-only, one request per trading day, cached
_cache_lock = threading.Lock()
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            with open(config.PH_CACHE_PATH) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _ph_series(ticker, s, e):
    today = dt.date.today()
    days = [s + dt.timedelta(d) for d in range((e - s).days + 1)]
    weekdays = [d for d in days if d.weekday() < 5 and d <= today]  # skip weekends / future
    if not weekdays:
        return None

    cache = _load_cache()
    results, to_fetch = {}, []
    for d in weekdays:
        key = f"{ticker}:{d.isoformat()}"
        if key in cache and d != today:   # never trust a cached "today" (still moving)
            results[d.isoformat()] = cache[key]
        else:
            to_fetch.append(d)

    if to_fetch:
        def fetch(d):
            return d.isoformat(), ph_data.get_ph_price_on(ticker, d.isoformat())
        with ThreadPoolExecutor(max_workers=config.PH_FETCH_WORKERS) as ex:
            for diso, val in ex.map(fetch, to_fetch):
                results[diso] = val
                if diso != today.isoformat():   # cache immutable past days (incl. None = no trade)
                    cache[f"{ticker}:{diso}"] = val
        with _cache_lock:
            try:
                with open(config.PH_CACHE_PATH, "w") as f:
                    json.dump(cache, f)
            except Exception:
                pass

    series = [{"date": d, "close": round(float(results[d]["close"]), 2),
               "volume": results[d].get("volume")}
              for d in sorted(results) if results[d]]
    if not series:
        return None
    return {"currency": "PHP", "has_ohlc": False, "series": series}


def daily_series(ticker: str, start_date: str, end_date: str) -> dict:
    """Normalised daily series for a US or PH ticker over [start, end] (inclusive)."""
    ticker = ticker.upper().strip()
    try:
        s = dt.date.fromisoformat(str(start_date).strip())
        e = dt.date.fromisoformat(str(end_date).strip())
    except ValueError as ex:
        return {"ok": False, "error": f"bad date: {ex}"}
    if e < s:
        return {"ok": False, "error": f"end date {e} is before start date {s}"}

    market = market_of(ticker)
    cap = config.PH_MAX_RANGE_DAYS if market == "PH" else _US_MAX_RANGE_DAYS
    if (e - s).days > cap:
        return {"ok": False, "error": f"date range too large (max {cap} days for {market})"}

    data = _ph_series(ticker, s, e) if market == "PH" else _us_series(ticker, s, e)
    if not data:
        return {"ok": False,
                "error": f"no price data for {ticker} in {start_date}..{end_date}"}
    return {"ok": True, "ticker": ticker, "market": market, **data}


def _ph_news(company_name: str, limit: int) -> list[dict]:
    """PH headlines via Google News RSS, keyed by company NAME (phisix and yfinance have no PH
    news; PH company news is reported by name, not ticker). Free, no API key. Returns the same
    shape as the US path. Google News RSS is unofficial, so any failure returns []."""
    q = urllib.parse.quote(f'"{company_name}"')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-PH&gl=PH&ceid=PH:en"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
    except Exception:
        return []
    items = []
    for it in root.findall(".//item")[:limit]:
        title = it.findtext("title")
        if not title:
            continue
        items.append({"title": title, "publisher": it.findtext("source"),
                      "link": it.findtext("link"), "published": it.findtext("pubDate")})
    return items


def get_recent_news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent headlines for a ticker. US goes through yfinance; PH has no ticker-based news
    feed (phisix/yfinance don't cover it), so we fall back to Google News RSS searched by the
    company name. Callers must treat an empty list as "no grounding available", not "nothing
    happened".
    """
    if market_of(ticker) == "PH":
        # PH news is reported by company name, not ticker - resolve it, then search Google News.
        name = ph_data.KNOWN_PSE.get(ticker.upper().strip(), ticker)
        return _ph_news(name, limit)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items = []
    for n in raw[:limit]:
        # yfinance has changed this payload shape across versions - some put fields at the top
        # level, newer ones nest under "content". Handle both rather than assume one.
        c = n.get("content", n) if isinstance(n, dict) else {}
        title = c.get("title")
        if not title:
            continue
        provider = c.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher")
        canonical = c.get("canonicalUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else c.get("link")
        published = c.get("pubDate") or n.get("providerPublishTime")
        items.append({"title": title, "publisher": publisher, "link": link, "published": published})
    return items


def current_price(ticker: str) -> dict:
    """Latest price for a US or PH ticker, routed to the right source."""
    ticker = ticker.upper().strip()
    if market_of(ticker) == "PH":
        r = ph_data.get_ph_current_price(ticker)
        if r.get("ok"):
            r["tool"] = "get_current_price"
        return r
    # US: last 5 trading days gives us the latest close + a prior close for the day move
    df = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
    if df.empty:
        return {"ok": False, "error": f"no recent price data for {ticker}"}
    close = df["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    return {
        "ok": True, "tool": "get_current_price", "market": "US", "ticker": ticker,
        "currency": "USD", "as_of": df.index[-1].date().isoformat(),
        "price": round(last, 2), "prev_close": round(prev, 2),
        "day_change_pct": round((last - prev) / prev * 100, 2) if prev else None,
        "series": [{"date": idx.date().isoformat(), "close": round(float(v), 2)}
                   for idx, v in close.items()],
    }