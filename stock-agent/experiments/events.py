"""FAANG events for the forecast-vetting dataset.

We build the test set around REAL price events. Rather than hand-guess earnings/launch dates, we
detect the biggest daily moves per FAANG ticker in 2026 straight from the price data - each big
move is a real event with a real, findable news catalyst around it. You can also hand-add known
events to CURATED (earnings, launches, macro shocks) and those take precedence.

The `event_date` is the day the move happened; the vetting harness forecasts as-of that date, so the
move itself lands inside the forecast horizon (the price-only model is blind to what caused it).
"""
import datetime as dt

from core import market_data

FAANG = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA"]
START = "2026-01-01"

# Optional hand-curated events. If non-empty, get_events() uses these instead of auto-detection.
# Shape: {"ticker","event_date" (YYYY-MM-DD),"event_type","description"}
CURATED: list[dict] = [
    # {"ticker": "NVDA", "event_date": "2026-02-26", "event_type": "earnings",
    #  "description": "Q4 FY26 earnings + guidance"},
]


def _top_moves(ticker: str, per_ticker: int, min_prior: int = 30, tail_pad: int = 6) -> list[dict]:
    """The `per_ticker` biggest daily % moves for a ticker in 2026, keeping only moves with enough
    prior history to fit a forecast and a few following days to measure the outcome."""
    today = dt.date.today().isoformat()
    ds = market_data.daily_series(ticker, START, today)
    if not ds.get("ok"):
        return []
    rows = ds["series"]
    moves = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1]["close"], rows[i]["close"]
        if not prev or i < min_prior or i > len(rows) - tail_pad:
            continue
        pct = (cur - prev) / prev * 100
        moves.append({
            "ticker": ticker,
            "event_date": rows[i]["date"],
            "event_type": "biggest_move",
            "move_pct": round(pct, 2),
            "description": f"{ticker} {'+' if pct >= 0 else ''}{round(pct, 2)}% on {rows[i]['date']}",
        })
    moves.sort(key=lambda m: abs(m["move_pct"]), reverse=True)
    return moves[:per_ticker]


def _calm_days(ticker: str, calm_per: int, min_prior: int = 30, tail_pad: int = 6) -> list[dict]:
    """The `calm_per` SMALLEST daily % moves for a ticker - control days with routine news, where a
    forecast should be reasonably trustworthy. These let us test that the agent DISCRIMINATES (says
    trust on calm days) instead of just always hedging 'caution' on the dramatic big-move days."""
    today = dt.date.today().isoformat()
    ds = market_data.daily_series(ticker, START, today)
    if not ds.get("ok"):
        return []
    rows = ds["series"]
    moves = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1]["close"], rows[i]["close"]
        if not prev or i < min_prior or i > len(rows) - tail_pad:
            continue
        pct = (cur - prev) / prev * 100
        moves.append({
            "ticker": ticker, "event_date": rows[i]["date"], "event_type": "calm_control",
            "move_pct": round(pct, 2),
            "description": f"{ticker} calm day {'+' if pct >= 0 else ''}{round(pct, 2)}% on {rows[i]['date']}",
        })
    moves.sort(key=lambda m: abs(m["move_pct"]))
    return moves[:calm_per]


def auto_events(big_per: int = 2, calm_per: int = 2) -> list[dict]:
    out = []
    for t in FAANG:
        out.extend(_top_moves(t, big_per))
        out.extend(_calm_days(t, calm_per))
    return out


def get_events(big_per: int = 2, calm_per: int = 2) -> list[dict]:
    """Hand-curated events if any, else auto-detected biggest-move days PLUS calm control days.
    The calm days are the control group for measuring discrimination (see eval_forecast_vetting)."""
    return CURATED if CURATED else auto_events(big_per, calm_per)
