"""Candidate-finder for the hand-curated benchmark (curated_events.csv).

This does NOT pick or label events - it only reads the 2026 price data and lists the biggest up/down
days per FAANG ticker (pure math, no AI), so you have a shortlist of dates to research by hand. For
each big day it also shows:
  gap%   - open vs previous close: a big gap means the news broke OVERNIGHT (look the evening before);
           a small gap with a big close move means it happened DURING the day (analyst note, mid-day news).
  SPY%   - the S&P 500's move that same day. If SPY moved almost as much in the same direction, the day
           was MARKET-WIDE (macro), not company-specific - be careful attributing it to one headline.

It also writes a starter CSV (move_candidates.csv) with ticker/move_date/direction pre-filled and the
rest blank, so you can copy rows straight into curated_events.csv and just add the news you find.

Run from the project root:
    ./.venv/Scripts/python.exe experiments/scan_moves.py           # top 8 movers/ticker + calm picks
    ./.venv/Scripts/python.exe experiments/scan_moves.py 12 4.0    # top 12, and mark >=4% as 'big'
"""
import csv
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import market_data  # noqa: E402

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
START = "2026-01-01"
OUT_TXT = ROOT / "experiments" / "results" / "move_candidates.txt"
OUT_CSV = ROOT / "experiments" / "data" / "curated" / "move_candidates.csv"
CSV_HEADER = ["ticker", "move_date", "direction", "label_type", "catalyst", "why",
              "expected_verdict", "headline_title", "headline_url", "headline_published", "notes"]


def _daily(ticker):
    """List of {date, pct, gap} for a ticker over 2026, pct/gap in percent vs the previous close."""
    end = dt.date.today().isoformat()
    ds = market_data.daily_series(ticker, START, end)
    if not ds["ok"]:
        print(f"[warn] {ticker}: {ds['error']}")
        return []
    rows, prev = [], None
    for r in ds["series"]:
        if prev is not None and prev:
            pct = (r["close"] - prev) / prev * 100
            gap = (r.get("open", r["close"]) - prev) / prev * 100
            rows.append({"date": r["date"], "pct": pct, "gap": gap})
        prev = r["close"]
    return rows


def run(top_n=8, big_thresh=4.0):
    spy = {d["date"]: d["pct"] for d in _daily("SPY")}  # market context

    L = [f"FAANG 2026 move candidates (from price data only - research the news yourself)",
         f"top {top_n} moves per ticker; gap% = overnight portion; SPY% = market that day", ""]
    csv_rows = []

    for tk in TICKERS:
        days = _daily(tk)
        if not days:
            continue
        movers = sorted(days, key=lambda d: abs(d["pct"]), reverse=True)[:top_n]
        movers.sort(key=lambda d: d["date"])
        L.append(f"=== {tk} ===")
        for d in movers:
            direction = "surge" if d["pct"] > 0 else "tank"
            sp = spy.get(d["date"])
            macro = ""
            if sp is not None and abs(sp) >= 2.0 and (sp > 0) == (d["pct"] > 0):
                macro = "  <- MARKET-WIDE day (check macro, not just the company)"
            when = "overnight" if abs(d["gap"]) >= abs(d["pct"]) * 0.6 else "intraday"
            sps = "n/a" if sp is None else f"{sp:+.1f}%"
            L.append(f"  {d['date']}  {d['pct']:+6.1f}%  ({direction:5}, {when:9})  "
                     f"gap {d['gap']:+5.1f}%  SPY {sps}{macro}")
            note = f"move {d['pct']:+.1f}%, gap {d['gap']:+.1f}%, SPY {sps}, {when}"
            csv_rows.append([tk, d["date"], direction, "", "", "", "", "", "", "", note])

        # a few CALM candidates: smallest-move days, spaced out, as control-day seeds
        calm = sorted(days, key=lambda d: abs(d["pct"]))[:3]
        calm.sort(key=lambda d: d["date"])
        L.append(f"  calm-day seeds (small moves, good controls): "
                 + ", ".join(f"{d['date']} ({d['pct']:+.1f}%)" for d in calm))
        for d in calm:
            csv_rows.append([tk, d["date"], "calm", "calm", "", "", "trust", "", "", "",
                             f"calm control, move {d['pct']:+.1f}%"])
        L.append("")

    report = "\n".join(L)
    print(report)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(report, encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(csv_rows)
    print(f"\nwrote {OUT_TXT}")
    print(f"wrote {OUT_CSV}  ({len(csv_rows)} candidate rows - copy the ones you research into "
          f"curated_events.csv and add the news)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    run(n, th)
