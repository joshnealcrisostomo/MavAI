"""Philippine Stock Exchange data source (phisix).

yfinance does not serve individual PSE stocks, so PH prices come from the free phisix API.
phisix gives a CURRENT snapshot only (price, volume, day change) - there is no free reliable
source of PSE daily history, so PH support is current-price only. See DECISIONS.md.

We try the api3 mirror first (the primary appspot instance is often cold / 503) and fall back.

Ticker coverage: KNOWN_PSE is built from phisix's own "/stocks.json" (ALL) endpoint instead of
a hand-picked list, so we cover whatever phisix itself currently tracks rather than a fixed
subset. The result is cached to disk with a TTL because:
  (a) the PSE ticker roster doesn't change intraday - a list built this morning is still valid
      this afternoon, so there's no need to hit the ALL endpoint on every request or every
      process start
  (b) if phisix is down/blocked when this module loads, we still want yesterday's list from
      cache rather than an empty one
  (c) if BOTH the live call and the cache fail (first run ever, offline dev, phisix fully dark),
      we fall back to a small hardcoded seed so the app doesn't stop recognising ANY PH stock

IMPORTANT CAVEAT (see DECISIONS.md): the official phisix-api.appspot.com service was shut down
in December 2023 following a takedown notice. The mirrors used here (api3, api2) are unofficial
and could go dark without warning - this module degrades gracefully (stale cache, then hardcoded
fallback) if that happens, but the underlying data source's long-term availability is not
guaranteed by us or by phisix's maintainers.
"""
import json
import os
import time

import requests

# phisix hosts, tried in order. Each returns {"stocks":[{name, price:{currency,amount},
# percentChange, volume, symbol}], "as_of": ...} for /stocks/<SYM>.json, or the full roster
# for /stocks.json (no symbol - the "ALL" resource).
_HOSTS = [
    "https://phisix-api3.appspot.com",
    "https://phisix-api.appspot.com",
    "https://phisix-api2.appspot.com",
]

# how long a cached ticker list is considered fresh before we try refetching live (hours)
_TICKER_CACHE_TTL_HOURS = float(os.getenv("PH_TICKER_CACHE_TTL_HOURS", "24"))
_TICKER_CACHE_PATH = os.getenv("PH_TICKER_CACHE_PATH", "./ph_tickers_cache.json")

# LAST-RESORT seed only - used if phisix is completely unreachable AND no cache exists yet
# (e.g. very first run, no network). This is no longer the coverage ceiling, just an emergency
# fallback so the app has *something* to route on before its first successful live fetch.
_FALLBACK_PSE = {
    "GLO": "Globe Telecom", "TEL": "PLDT", "BDO": "BDO Unibank", "BPI": "Bank of the PH Islands",
    "SM": "SM Investments", "SMPH": "SM Prime Holdings", "AC": "Ayala Corporation",
    "ALI": "Ayala Land", "JFC": "Jollibee Foods", "MER": "Meralco", "URC": "Universal Robina",
    "ICT": "Intl Container Terminal", "AEV": "Aboitiz Equity", "JGS": "JG Summit",
    "MBT": "Metrobank", "PGOLD": "Puregold", "DMC": "DMCI Holdings", "GTCAP": "GT Capital",
}


def _fetch_all_pse_live():
    """Hit phisix's ALL-stocks resource (/stocks.json, no symbol) across hosts in order.

    Returns {SYMBOL: company_name} for every stock phisix currently tracks, or None if every
    host failed (network error, non-200, or unexpected payload shape).
    """
    for host in _HOSTS:
        try:
            resp = requests.get(f"{host}/stocks.json", timeout=20)
        except requests.RequestException:
            continue
        if resp.status_code != 200:
            continue  # cold instance / blocked / down - try the next host
        try:
            stocks = resp.json()["stocks"]
        except (ValueError, KeyError, TypeError):
            continue
        result = {}
        for s in stocks:
            sym = s.get("symbol") if isinstance(s, dict) else None
            if sym:
                result[sym.upper().strip()] = s.get("name", sym)
        if result:
            return result
    return None


def _read_ticker_cache():
    """Returns (ticker_dict_or_None, age_in_hours). age is inf if there's no usable cache."""
    try:
        with open(_TICKER_CACHE_PATH) as f:
            payload = json.load(f)
        age_hours = (time.time() - payload.get("fetched_at", 0)) / 3600
        return payload.get("tickers"), age_hours
    except Exception:
        return None, float("inf")


def _write_ticker_cache(tickers: dict):
    try:
        with open(_TICKER_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": time.time(), "tickers": tickers}, f)
    except Exception:
        pass  # caching is best-effort - a failed write must never break the request


def _load_known_pse() -> dict:
    """Resolution order: fresh cache -> live fetch -> stale cache -> hardcoded fallback.

    This is what makes KNOWN_PSE reflect the full exchange phisix tracks, refreshed
    periodically, instead of a fixed hand-picked list frozen at whatever we typed in once.
    """
    cached, age_hours = _read_ticker_cache()
    if cached and age_hours < _TICKER_CACHE_TTL_HOURS:
        return cached

    live = _fetch_all_pse_live()
    if live:
        _write_ticker_cache(live)
        return live

    if cached:  # live fetch failed, but a (stale) cache beats nothing
        return cached

    return dict(_FALLBACK_PSE)  # first run ever, no network at all - last resort only


def refresh_known_pse() -> dict:
    """Force a live refetch, ignoring cache freshness, and update KNOWN_PSE IN PLACE.

    Uses .clear()/.update() rather than reassigning the name, so other modules that reference
    ph_data.KNOWN_PSE as a live attribute (e.g. market_data.market_of()'s membership check) see
    the refreshed roster immediately, without needing to re-import anything. Call this from a
    scheduled job / admin endpoint if you want periodic refreshes beyond the TTL-on-read check.
    """
    live = _fetch_all_pse_live()
    if live:
        _write_ticker_cache(live)
        KNOWN_PSE.clear()
        KNOWN_PSE.update(live)
    return KNOWN_PSE


# built once at import time; see _load_known_pse for the fallback chain
KNOWN_PSE = _load_known_pse()


def get_ph_current_price(symbol: str) -> dict:
    """Current price of one PSE stock from phisix. Shape mirrors the US current-price tool."""
    symbol = symbol.upper().strip()
    last_err = "no response"
    for host in _HOSTS:
        try:
            resp = requests.get(f"{host}/stocks/{symbol}.json", timeout=15)
        except requests.RequestException as e:
            last_err = f"request failed: {e}"
            continue
        if resp.status_code == 200:
            try:
                stock = resp.json()["stocks"][0]
            except (ValueError, KeyError, IndexError):
                last_err = "unexpected response format"
                continue
            price = stock.get("price", {})
            return {
                "ok": True,
                "tool": "get_ph_current_price",
                "market": "PSE",
                "ticker": symbol,
                "name": stock.get("name"),
                "price": price.get("amount"),
                "currency": price.get("currency", "PHP"),
                "day_change_pct": stock.get("percentChange"),
                "volume": stock.get("volume"),
            }
        # 500 = phisix's "unknown symbol"; 503 = cold instance, try the next host
        if resp.status_code == 500:
            return {"ok": False, "error": f"'{symbol}' is not a recognised PSE ticker"}
        last_err = f"HTTP {resp.status_code}"
    return {"ok": False, "error": f"PH data source unavailable ({last_err})"}


def get_ph_price_on(symbol: str, date_iso: str):
    """Closing price + volume for one PSE stock on one date, or None if no trading that day.

    phisix exposes a per-date endpoint (/stocks/<SYM>.<YYYY-MM-DD>.json). A 200 is a trading
    day; 404/500 means no data for that date (weekend/holiday) or unknown symbol; 503 is a cold
    host, so we try the next one.
    """
    symbol = symbol.upper().strip()
    for host in _HOSTS:
        try:
            resp = requests.get(f"{host}/stocks/{symbol}.{date_iso}.json", timeout=15)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            try:
                stock = resp.json()["stocks"][0]
                return {"close": float(stock["price"]["amount"]), "volume": stock.get("volume")}
            except (ValueError, KeyError, IndexError):
                return None
        if resp.status_code == 503:
            continue  # cold host, try the next
        return None   # 404/500 -> no data for that day
    return None