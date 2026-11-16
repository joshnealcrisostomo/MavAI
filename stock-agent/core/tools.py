"""Tool Use module: the functions the ReAct agent can call.

The tools are market-agnostic. Each one gets its daily prices from core.market_data, which
routes to yfinance (US) or phisix (PH) and hands back a normalised series of closes. So the
same min/max/average and moving-average logic works for both markets - the agent just names a
ticker and doesn't have to care where the data comes from.

Each tool returns compact stats (fed back to the model) plus a chart series (kept for the UI,
stripped before it goes back into the model context). Only US data carries full OHLC, so only
US stocks can be drawn as candlesticks; PH is close-only (line charts, moving averages, stats).
"""
import datetime as dt
import logging
from email.utils import parsedate_to_datetime

from . import config, forecast, market_data, news_hist


def _parse_news_date(published):
    """Best-effort parse of a headline's 'published' field to a date. Handles RFC822 (Google News
    RSS), ISO (yfinance pubDate), and epoch seconds (older yfinance)."""
    if not published:
        return None
    if isinstance(published, (int, float)):
        try:
            return dt.datetime.fromtimestamp(published, dt.timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    s = str(published).strip()
    try:
        return parsedate_to_datetime(s).date()
    except (TypeError, ValueError):
        pass
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def _biggest_move(series):
    """Largest day-over-day % move in a close series: {date, pct, prev_close, close} or None."""
    if len(series) < 2:
        return None
    best = None
    for i in range(1, len(series)):
        prev, cur = series[i - 1]["close"], series[i]["close"]
        if not prev:
            continue
        pct = (cur - prev) / prev * 100
        if best is None or abs(pct) > abs(best["pct"]):
            best = {"date": series[i]["date"], "pct": round(pct, 2),
                    "prev_close": prev, "close": cur}
    return best

# yfinance logs failed lookups to stderr; we return clean error dicts, so quiet it down.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def get_stock_history(ticker: str, start_date: str, end_date: str) -> dict:
    """Min / max / average / first / last closing price over a date range, plus the series.
    Works for US and Philippine (PSE) tickers."""
    ds = market_data.daily_series(ticker, start_date, end_date)
    if not ds["ok"]:
        return {"ok": False, "error": ds["error"]}
    series = ds["series"]
    closes = [r["close"] for r in series]
    first, last = closes[0], closes[-1]
    return {
        "ok": True,
        "tool": "get_stock_history",
        "ticker": ds["ticker"],
        "market": ds["market"],
        "currency": ds["currency"],
        "start_date": start_date,
        "end_date": end_date,
        "trading_days": len(series),
        "min_close": round(min(closes), 2),
        "max_close": round(max(closes), 2),
        "avg_close": round(sum(closes) / len(closes), 2),
        "first_close": round(first, 2),
        "last_close": round(last, 2),
        "pct_change": round((last - first) / first * 100, 2) if first else None,
        "has_ohlc": ds["has_ohlc"],
        "series": series,
    }


def get_current_price(ticker: str) -> dict:
    """Latest available price and day-over-day change for a US or PSE ticker."""
    return market_data.current_price(ticker)


def get_moving_average(ticker: str, start_date: str, end_date: str, window: int = 5) -> dict:
    """Simple moving average of the close over a range, for a US or PSE ticker.
    Returns both the MA series and the underlying price series for the chart overlay."""
    try:
        window = int(window)
    except (TypeError, ValueError):
        return {"ok": False, "error": "window must be an integer"}
    if window < 2:
        return {"ok": False, "error": "window must be >= 2"}
    try:
        s = dt.date.fromisoformat(str(start_date).strip())
    except ValueError as e:
        return {"ok": False, "error": f"bad start_date: {e}"}

    # pull extra days before the start so the first in-range MA point is already warmed up
    pad_start = (s - dt.timedelta(days=window * 3 + 10)).isoformat()
    ds = market_data.daily_series(ticker, pad_start, end_date)
    if not ds["ok"]:
        return {"ok": False, "error": ds["error"]}

    rows = ds["series"]  # sorted ascending by date
    vals = [r["close"] for r in rows]
    ma_series = []
    for i in range(window - 1, len(vals)):
        avg = sum(vals[i - window + 1:i + 1]) / window
        if rows[i]["date"] >= start_date:
            ma_series.append({"date": rows[i]["date"], "ma": round(avg, 2)})
    if not ma_series:
        return {"ok": False, "error": "not enough data to compute the moving average"}
    price_series = [{"date": r["date"], "close": r["close"]}
                    for r in rows if r["date"] >= start_date]
    return {
        "ok": True,
        "tool": "get_moving_average",
        "ticker": ds["ticker"],
        "market": ds["market"],
        "currency": ds["currency"],
        "window": window,
        "start_date": start_date,
        "end_date": end_date,
        "latest_ma": ma_series[-1]["ma"],
        "ma_series": ma_series,
        "series": price_series,
    }


def get_price_move_context(ticker: str, start_date: str, end_date: str) -> dict:
    """Price move over a range plus the news around it, for grounding a 'why did it move' answer.
    This does NOT explain the move itself - it hands back real numbers and real headlines so the
    model reasons over them instead of inventing a cause.

    News comes from the unified news layer (news_hist: Finnhub/GDELT for US, Google News for PH),
    fetched over a window AROUND the move date. Because this is retrospective explanation, not
    prediction, we do NOT apply a no-lookahead cut - same-day and intraday news is exactly what
    usually explains a past move. Each headline is tagged before_move: a headline can plausibly
    explain a move only if it was published on or before the move day, so later recaps are flagged
    before_move=false (kept as context, but not to be cited as the cause).
    """
    # Fetch a few days BEFORE the requested start too, so a single-day query (start == end) still
    # has a prior close to measure the day's move against - otherwise the move computes as 0%.
    try:
        s_date = dt.date.fromisoformat(str(start_date).strip())
        e_date = dt.date.fromisoformat(str(end_date).strip())
    except ValueError as ex:
        return {"ok": False, "error": f"bad date: {ex}"}
    ds = market_data.daily_series(ticker, (s_date - dt.timedelta(days=7)).isoformat(),
                                  end_date)
    if not ds["ok"]:
        return {"ok": False, "error": ds["error"]}
    full = ds["series"]
    user = [r for r in full if start_date <= r["date"] <= end_date]
    if not user:
        return {"ok": False, "error": f"no price data for {ds['ticker']} in {start_date}..{end_date}"}
    first, last = user[0]["close"], user[-1]["close"]
    pct = round((last - first) / first * 100, 2) if first else None

    # biggest day-over-day move whose DAY falls in the requested window (uses the prior close, which
    # may come from the pre-window buffer - this is what makes single-day queries work).
    move, user_dates = None, {r["date"] for r in user}
    for i in range(1, len(full)):
        if full[i]["date"] not in user_dates:
            continue
        prev, cur = full[i - 1]["close"], full[i]["close"]
        if not prev:
            continue
        m = (cur - prev) / prev * 100
        if move is None or abs(m) > abs(move["pct"]):
            move = {"date": full[i]["date"], "pct": round(m, 2), "prev_close": prev, "close": cur}

    direction = "up" if (pct or 0) > 0 else "down" if (pct or 0) < 0 else "flat"

    # news AROUND the move window: a few days before the start through the day after the end (recaps),
    # no open cut. A headline "could explain" the move if it was published on or before the END of the
    # requested window - so same-window and same-day/intraday news counts; only later recaps don't.
    # Fetch a LARGE batch (the source returns newest-first, so a small cap would crowd out the
    # mid-window news that actually explains the move), then keep the 15 headlines NEAREST the move
    # day rather than the 15 newest - otherwise a wide window keeps only end-of-range items.
    # Anchor the news fetch to the MOVE day, not the whole requested range: a heavily-covered
    # mega-cap can have >400 articles across a two-week window, and the source returns newest-first,
    # so a wide fetch would crowd out the days that actually explain the move. A tight window around
    # the move (same 3-day lookback vet_forecast uses) reliably surfaces the catalyst.
    move_ref = dt.date.fromisoformat(move["date"]) if move else e_date
    raw = news_hist.news_for_ticker(ds["ticker"], (move_ref - dt.timedelta(days=3)).isoformat(),
                                    (move_ref + dt.timedelta(days=1)).isoformat(), max_records=400)
    news = []
    for n in raw:
        d = _parse_news_date(n.get("seendate"))
        news.append({"title": n.get("title"), "publisher": n.get("domain"), "link": n.get("url"),
                     "published": n.get("seendate"), "date": d.isoformat() if d else None,
                     "before_move": bool(d and d <= e_date), "_d": d})
    news.sort(key=lambda x: (abs((x["_d"] - move_ref).days) if x["_d"] else 9999,
                             x.get("date") or ""))
    news = news[:15]
    for n in news:
        n.pop("_d", None)

    return {
        "ok": True,
        "tool": "get_price_move_context",
        "ticker": ds["ticker"],
        "market": ds["market"],
        "currency": ds["currency"],
        "start_date": start_date,
        "end_date": end_date,
        "first_close": round(first, 2),
        "last_close": round(last, 2),
        "pct_change": pct,
        "direction": direction,   # up/down/flat over the requested window (state this, don't echo the user)
        "biggest_move": move,     # {date, pct, prev_close, close} or None
        "has_news": bool(news),
        "news": news,
        "note": ("Report the move using direction/pct_change - do NOT assume the user's stated "
                 "direction. Attribute the move only from headlines with before_move=true, as "
                 "correlation not proven causation. If no before_move headline fits, say no news in "
                 "our window explains it."),
        "series": user,
    }


def get_volatility(ticker: str, start_date: str, end_date: str) -> dict:
    """Standard deviation of daily returns over a range - how much a stock's price swings day
    to day, the same definition the glossary in rag.py already gives. Reports both the raw
    daily figure and the annualized figure (daily * sqrt(252) trading days/year), which is the
    number usually quoted. Works for US and PH tickers - it only needs a close series, same as
    the other tools, no separate data source needed."""
    ds = market_data.daily_series(ticker, start_date, end_date)
    if not ds["ok"]:
        return {"ok": False, "error": ds["error"]}
    closes = [r["close"] for r in ds["series"]]
    if len(closes) < 3:
        return {"ok": False, "error": "need at least 3 trading days to compute volatility"}

    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1]]
    if len(returns) < 2:
        return {"ok": False, "error": "not enough price changes to compute volatility"}

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)  # sample stdev
    daily_std = variance ** 0.5
    annualized = daily_std * (252 ** 0.5)  # ~252 trading days/year, the standard convention

    returns_series = [{"date": ds["series"][i]["date"], "return_pct": round(returns[i - 1] * 100, 3)}
                       for i in range(1, len(ds["series"]))]
    return {
        "ok": True,
        "tool": "get_volatility",
        "ticker": ds["ticker"],
        "market": ds["market"],
        "currency": ds["currency"],
        "start_date": start_date,
        "end_date": end_date,
        "trading_days": len(closes),
        "daily_volatility_pct": round(daily_std * 100, 3),
        "annualized_volatility_pct": round(annualized * 100, 2),
        "returns_series": returns_series,
        "series": ds["series"],
    }


def get_correlation(ticker_a: str, ticker_b: str, start_date: str, end_date: str) -> dict:
    """Pearson correlation of daily returns between two stocks over the same range - how much
    they tend to move together (+1 perfectly together, -1 perfectly opposite, 0 unrelated).
    Works across any mix of US and PH tickers because it correlates % returns, not price
    level, same reasoning as compare_stocks. This needs two full series aligned on shared
    trading days, which is exactly the kind of multi-fetch-and-compute task a single search
    result can't answer in one query."""
    ds_a = market_data.daily_series(ticker_a, start_date, end_date)
    if not ds_a["ok"]:
        return {"ok": False, "error": f"{ticker_a}: {ds_a['error']}"}
    ds_b = market_data.daily_series(ticker_b, start_date, end_date)
    if not ds_b["ok"]:
        return {"ok": False, "error": f"{ticker_b}: {ds_b['error']}"}

    closes_a = {r["date"]: r["close"] for r in ds_a["series"]}
    closes_b = {r["date"]: r["close"] for r in ds_b["series"]}
    # only dates BOTH markets traded on - a US holiday that isn't a PH holiday (or vice versa)
    # would otherwise misalign the two return series against each other
    common_dates = sorted(set(closes_a) & set(closes_b))
    if len(common_dates) < 4:
        return {"ok": False, "error": "not enough overlapping trading days to compute correlation"}

    returns_a, returns_b = [], []
    for i in range(1, len(common_dates)):
        d0, d1 = common_dates[i - 1], common_dates[i]
        pa0, pb0 = closes_a[d0], closes_b[d0]
        if pa0 and pb0:
            returns_a.append((closes_a[d1] - pa0) / pa0)
            returns_b.append((closes_b[d1] - pb0) / pb0)
    n = len(returns_a)
    if n < 3:
        return {"ok": False, "error": "not enough overlapping price changes to compute correlation"}

    mean_a, mean_b = sum(returns_a) / n, sum(returns_b) / n
    cov = sum((returns_a[i] - mean_a) * (returns_b[i] - mean_b) for i in range(n)) / n
    std_a = (sum((r - mean_a) ** 2 for r in returns_a) / n) ** 0.5
    std_b = (sum((r - mean_b) ** 2 for r in returns_b) / n) ** 0.5
    if std_a == 0 or std_b == 0:
        return {"ok": False, "error": "one of the stocks had zero price variation over this range"}
    correlation = cov / (std_a * std_b)

    scatter = [{"date": common_dates[i + 1],
                "return_a_pct": round(returns_a[i] * 100, 3),
                "return_b_pct": round(returns_b[i] * 100, 3)} for i in range(n)]
    return {
        "ok": True,
        "tool": "get_correlation",
        "ticker_a": ds_a["ticker"], "ticker_b": ds_b["ticker"],
        "market_a": ds_a["market"], "market_b": ds_b["market"],
        "start_date": start_date, "end_date": end_date,
        "overlapping_days": n,
        "correlation": round(correlation, 3),
        "scatter": scatter,
    }


def compare_stocks(tickers, start_date: str, end_date: str) -> dict:
    """Compare total return of 2-5 tickers over the same range, normalised to 0% at the start.
    Works across US and PH because it compares percentage return, not price level."""
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace(" ", ",").split(",") if t.strip()]
    tickers = [t.upper().strip() for t in tickers][:5]  # cap at 5 for a readable chart
    if len(tickers) < 2:
        return {"ok": False, "error": "need at least 2 tickers to compare"}

    results, series, markets = {}, {}, {}
    for t in tickers:
        ds = market_data.daily_series(t, start_date, end_date)
        if not ds["ok"]:
            results[t] = {"error": ds["error"]}
            continue
        closes = [r["close"] for r in ds["series"]]
        first, last = closes[0], closes[-1]
        markets[t] = ds["market"]
        results[t] = {
            "first_close": round(first, 2),
            "last_close": round(last, 2),
            "pct_change": round((last - first) / first * 100, 2) if first else None,
        }
        series[t] = [{"date": r["date"], "pct": round((r["close"] - first) / first * 100, 2)}
                     for r in ds["series"]]

    ranked = [t for t in results if results[t].get("pct_change") is not None]
    
    if len(ranked) < 2:
        return {
            "ok": False, 
            "error": "Need at least 2 valid tickers with price data to compare."
        }
        
    winner = max(ranked, key=lambda t: results[t]["pct_change"], default=None)
    return {
        "ok": True,
        "tool": "compare_stocks",
        "tickers": tickers,
        "markets": markets,
        "start_date": start_date,
        "end_date": end_date,
        "results": results,
        "best_performer": winner,
        "comparison_series": series,
    }


def vet_forecast(ticker: str, as_of_date: str = None, horizon: int = None) -> dict:
    """Bundle a price-only forecast + the news available BEFORE a date + (in backtest) the actual
    outcome, so the agent can VET whether the forecast is trustworthy given news the model never
    saw. This is the Component #14 (DS model) integration. It does NOT predict the price or give
    advice - it produces a forecast to CRITIQUE.

    as_of_date is the day the forecast is made 'as of': the model sees only closes strictly before
    it (defaults to today for live use). US (FAANG/Big Tech) tickers only. When as_of is in the
    past, the real next-`horizon`-day outcome is attached for scoring; when it's ~today, `actual`
    is unavailable (nothing to score yet)."""
    ticker = ticker.upper().strip()
    if market_data.market_of(ticker) != "US":
        return {"ok": False, "error": f"vet_forecast supports US tickers only, not {ticker}"}
    horizon = int(horizon) if horizon else config.FORECAST_HORIZON
    if not as_of_date:
        as_of = dt.date.today()
    else:
        try:
            as_of = dt.date.fromisoformat(str(as_of_date).strip())
        except ValueError as e:
            return {"ok": False, "error": f"bad as_of_date: {e}"}
    as_of_iso = as_of.isoformat()

    # ~180 calendar days (~120 trading days) of history before the event, plus a little past it
    # so a backtest can read the realised outcome. daily_series is inclusive of both ends.
    start = (as_of - dt.timedelta(days=180)).isoformat()
    end = (as_of + dt.timedelta(days=horizon + 12)).isoformat()
    ds = market_data.daily_series(ticker, start, end)
    if not ds["ok"]:
        return {"ok": False, "error": ds["error"]}

    series = ds["series"]
    in_sample = [r for r in series if r["date"] < as_of_iso]   # only what the model may see
    after = [r for r in series if r["date"] >= as_of_iso]
    if len(in_sample) < 8:
        return {"ok": False, "error": f"not enough pre-{as_of_iso} history for {ticker}"}

    fc = forecast.forecast(in_sample, horizon=horizon, backend="auto", as_of=as_of_iso)
    base = forecast.naive_forecast(in_sample, horizon=horizon, as_of=as_of_iso)

    # realised outcome over the horizon - backtest only; empty when as_of is ~today
    horizon_rows = after[:horizon]
    last_in = in_sample[-1]["close"]
    actual = {"available": False}
    if horizon_rows:
        end_close = horizon_rows[-1]["close"]
        actual = {
            "available": True,
            "series": [{"date": r["date"], "close": r["close"]} for r in horizon_rows],
            "end_close": round(end_close, 2),
            "realized_pct": round((end_close - last_in) / last_in * 100, 2) if last_in else None,
            "direction": "up" if end_close > last_in else "down" if end_close < last_in else "flat",
        }

    # Overnight / pre-open news window. We fetch through the event day itself and then cut by
    # TIMESTAMP at the move-day market open (~09:30 ET), so news_before includes the overnight and
    # pre-market catalyst (the usual driver of the open gap) but never anything published once the
    # session - and the move - is under way. That timestamp cut is the whole point: a coarse
    # "prior days only" window threw the actual catalyst away with the rest of the event day.
    # fetch a big slice: Finnhub returns newest-first, so the event-day evening items would
    # otherwise crowd out the overnight/pre-open ones we actually want before the timestamp cut.
    w_start = (as_of - dt.timedelta(days=3)).isoformat()
    fetched = news_hist.news_for_ticker(ticker, w_start, as_of_iso, max_records=400)
    open_cut = dt.datetime(as_of.year, as_of.month, as_of.day, 13, 30,
                           tzinfo=dt.timezone.utc).timestamp()   # ~09:30 ET (conservative vs DST)

    def _before_open(n):
        if n.get("ts") is not None:
            return n["ts"] < open_cut                    # precise: strictly before the open
        return (n.get("seendate") or "") < as_of_iso     # date-only source: keep prior days only

    news_before = [n for n in fetched if _before_open(n)]
    news_before = sorted(news_before, key=lambda n: (n.get("ts") or 0, n.get("seendate") or ""),
                         reverse=True)[:15]

    return {
        "ok": True,
        "tool": "vet_forecast",
        "ticker": ds["ticker"],
        "market": ds["market"],
        "currency": ds["currency"],
        "as_of_date": as_of_iso,
        "horizon": horizon,
        # DIRECTION only for the model - the numeric point/range live in forecast_series (a _SERIES_KEYS
        # key, stripped from the model's view) so the agent cannot quote a predicted price/target. We
        # vet reliability, we do not predict a price.
        "forecast": {"ok": fc.get("ok"), "model": fc.get("model"), "horizon": horizon,
                     "as_of": as_of_iso, "direction": fc.get("direction"),
                     "last_close": fc.get("last_close")},
        "baseline": {"direction": base.get("direction")},
        "forecast_series": fc.get("predictions"),   # numbers kept for the UI, hidden from the model
        "news_before": news_before,
        "actual": actual,
        "note": ("This critiques a price-only forecasting ALGORITHM against news it could not see. "
                 "It is not investment advice, a price prediction, or a price target. Cite only "
                 "headlines from news_before; if the news implies a shock the model can't "
                 "anticipate, the forecast is less trustworthy."),
    }


# NOTE: FinBERT sentiment (core/sentiment.py) was evaluated but NOT shipped as a live tool - it did
# not generalise out-of-domain (98% on PhraseBank -> 45% on FiQA; the general LLM read our headlines
# better). Kept as an RRL finding; see experiments/RRL-FINDINGS.md and eval_finbert*.py.


# OpenAI-style tool schemas handed to Groq. Descriptions say both markets are supported so the
# model calls the same tool for any stock; it only needs to resolve the company to its ticker.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "vet_forecast",
            "description": "Assess how RELIABLE a price-only statistical forecast is for a US stock, "
                           "given the recent news the forecasting model could not see. Use ONLY when "
                           "the user asks about the trustworthiness/reliability of a forecast or model "
                           "prediction (e.g. 'how reliable is the forecast for Tesla', 'should I trust "
                           "the model's forecast for AAPL', 'is the forecast likely to be wrong'). Do "
                           "NOT use it for ordinary price/history/compare questions. This is an "
                           "assessment of an ALGORITHM's reliability, never a price prediction or "
                           "buy/sell advice. US FAANG/Big Tech tickers only (AAPL, MSFT, GOOG, AMZN, "
                           "META, NVDA, TSLA).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US ticker, e.g. TSLA or AAPL"},
                    "as_of_date": {"type": "string",
                                   "description": "optional YYYY-MM-DD to assess as of a past date; omit for now"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_history",
            "description": "Min, max, average, first and last closing price for one stock over a "
                           "date range. Works for US tickers (AAPL, MSFT) and Philippine PSE "
                           "tickers (GLO, TEL, BDO, JFC). Use for 'min/max/average price' questions "
                           "AND for any past period like 'last month', 'in Q1', or 'over the last 3 "
                           "months' - convert the period to a start_date and end_date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US or PSE ticker, e.g. AAPL or GLO"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_price",
            "description": "Latest/current price and day-over-day change for one stock (US or PSE). "
                           "Use ONLY for the current/latest/now price. For any PAST date or period "
                           "(e.g. 'last month', 'in Q1', a date range) use get_stock_history instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US or PSE ticker, e.g. AAPL or GLO"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_moving_average",
            "description": "Simple moving average of the closing price over a range. Works for US "
                           "and Philippine PSE tickers. Use for 'N-day moving average' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US or PSE ticker, e.g. MSFT or TEL"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "window": {"type": "integer", "description": "MA window in trading days, e.g. 5, 20, 50"},
                },
                "required": ["ticker", "start_date", "end_date", "window"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_move_context",
            "description": "Get the price change over a range PLUS recent news headlines and the "
                           "biggest daily move, to ground a 'why' question (why did X drop/rise, "
                           "what happened to X, what's going on with X). Use this instead of "
                           "get_stock_history whenever the user asks for a REASON or EXPLANATION "
                           "for a move. Works for US (yfinance news) and PH (Google News). Each "
                           "headline is tagged before_move; attribute the move ONLY from headlines "
                           "dated on or before it, and as correlation, not proven causation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US or PSE ticker, e.g. AAPL or GLO"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_volatility",
            "description": "Standard deviation of daily returns over a range - how much a "
                           "stock's price swings day to day. Reports both the raw daily figure "
                           "and the annualized figure (the number usually quoted). Works for US "
                           "and PSE tickers. Use for 'volatility', 'how risky/choppy is X', or "
                           "'how much does X swing' questions - this is descriptive risk data, "
                           "not a recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "US or PSE ticker, e.g. AAPL or GLO"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation",
            "description": "Pearson correlation of daily returns between two stocks over the "
                           "same range - how much they move together (+1 in lockstep, -1 "
                           "opposite, 0 unrelated). Works for any mix of US and PSE tickers. Use "
                           "for 'how correlated are X and Y', 'do X and Y move together', or "
                           "'would X and Y both drop at once' questions - report the number and "
                           "explain what it means, do not recommend holding or avoiding either.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_a": {"type": "string", "description": "First US or PSE ticker"},
                    "ticker_b": {"type": "string", "description": "Second US or PSE ticker"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker_a", "ticker_b", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare total return of 2-5 stocks over the same date range. Works "
                           "for US and Philippine PSE tickers. Use for 'compare X and Y' or "
                           "'which performed better' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"},
                                "description": "2 to 5 US or PSE tickers, e.g. ['AAPL','MSFT'] or ['GLO','TEL']"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["tickers", "start_date", "end_date"],
            },
        },
    },
]

# name -> callable, used by the agent loop to dispatch a tool call
TOOL_FUNCS = {
    "get_stock_history": get_stock_history,
    "get_current_price": get_current_price,
    "get_moving_average": get_moving_average,
    "get_price_move_context": get_price_move_context,
    "get_volatility": get_volatility,
    "get_correlation": get_correlation,
    "compare_stocks": compare_stocks,
    "vet_forecast": vet_forecast,
}

# keys that hold the big chart payloads - stripped before a tool result goes back to the model
_SERIES_KEYS = ("series", "ma_series", "comparison_series", "returns_series", "scatter", "headlines",
                "forecast_series")


def run_tool(name: str, args: dict) -> dict:
    """Dispatch one tool call. Returns the full result dict (including chart series)."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # never let a tool crash take down the request
        return {"ok": False, "error": f"{name} failed: {e}"}


def compact(result: dict) -> dict:
    """Strip the heavy chart series so only small stats go back into the model context."""
    return {k: v for k, v in result.items() if k not in _SERIES_KEYS}