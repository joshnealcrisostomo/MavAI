"""
verify_candidates.py  (fixed)

Double-checks the move / gap / SPY numbers in candidates.csv against real
historical price data from Yahoo Finance (via yfinance).

FIX vs. previous version: the old script cached only a +/-7 day window
around the FIRST date seen per ticker, so every later date for that ticker
(often months away) fell outside the window and wrongly flagged as
"not a trading day." This version pulls each ticker's full date range
(Jan 1 - Aug 8, 2026, with a few days padding) ONCE, so every row is
checked correctly.

USAGE
-----
1. Install the dependency (one time):
       pip install yfinance pandas

2. Put your candidates in candidates.csv next to this script, with columns:
       ticker,move_date,direction,move_pct,gap_pct,spy_pct,timing

3. Run:
       python verify_candidates.py

4. Read verification_report.csv — it shows, per row:
       - the move/gap % your CSV claims
       - the move/gap % actually computed from Yahoo data
       - a PASS/FLAG status (FLAG = off by more than the tolerance below)
"""

import pandas as pd
import yfinance as yf

# ---- settings you can tweak -------------------------------------------------
CANDIDATES_FILE = "candidates.csv"
REPORT_FILE = "verification_report.csv"
TOLERANCE_PCT = 0.5        # a mismatch bigger than this (percentage points) is flagged
RANGE_START = "2025-12-20"  # a bit before Jan 1 so the very first trading day has a prior close
RANGE_END = "2026-08-10"    # a bit after Aug 8 so Aug 8 data is fully included
# ------------------------------------------------------------------------------


def pct_change(open_, prev_close, close):
    """Return (move_pct, gap_pct) as percentages."""
    move = (close - prev_close) / prev_close * 100
    gap = (open_ - prev_close) / prev_close * 100
    return move, gap


def get_full_history(ticker):
    """Download the FULL Jan-Aug 2026 daily OHLC range for `ticker`, once."""
    df = yf.download(ticker, start=RANGE_START, end=RANGE_END,
                      progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def find_prev_trading_day(df, date):
    """Given a DataFrame indexed by trading date, find the row before `date`."""
    prior = df.index[df.index < pd.Timestamp(date)]
    if len(prior) == 0:
        return None
    return prior.max()


def verify_row(row, spy_df, ticker_df):
    ticker = row["ticker"]
    date = pd.to_datetime(row["move_date"])

    result = {
        "ticker": ticker,
        "move_date": row["move_date"],
        "direction": row.get("direction"),
        "claimed_move_pct": row.get("move_pct"),
        "claimed_gap_pct": row.get("gap_pct"),
        "claimed_spy_pct": row.get("spy_pct"),
        "timing": row.get("timing"),
    }

    try:
        if pd.Timestamp(date) not in ticker_df.index:
            result["status"] = "FLAG"
            result["issue"] = "move_date not found in ticker trading data (check date/ticker)"
            return result

        prev_day = find_prev_trading_day(ticker_df, date)
        if prev_day is None:
            result["status"] = "FLAG"
            result["issue"] = "no prior trading day found before this date"
            return result

        close_today = ticker_df.loc[pd.Timestamp(date), "Close"]
        open_today = ticker_df.loc[pd.Timestamp(date), "Open"]
        close_prev = ticker_df.loc[prev_day, "Close"]

        actual_move, actual_gap = pct_change(open_today, close_prev, close_today)

        # SPY same-day move
        spy_prev = find_prev_trading_day(spy_df, date)
        if pd.Timestamp(date) in spy_df.index and spy_prev is not None:
            spy_close_today = spy_df.loc[pd.Timestamp(date), "Close"]
            spy_close_prev = spy_df.loc[spy_prev, "Close"]
            actual_spy = (spy_close_today - spy_close_prev) / spy_close_prev * 100
        else:
            actual_spy = None

        result["actual_move_pct"] = round(float(actual_move), 2)
        result["actual_gap_pct"] = round(float(actual_gap), 2)
        result["actual_spy_pct"] = round(float(actual_spy), 2) if actual_spy is not None else None
        result["prev_trading_day"] = prev_day.strftime("%Y-%m-%d")

        # --- compare vs claimed, where a claimed value exists ---
        issues = []
        claimed_move = row.get("move_pct")
        if pd.notna(claimed_move):
            diff = abs(actual_move - float(claimed_move))
            if diff > TOLERANCE_PCT:
                issues.append(f"move off by {diff:.2f}pp")

        claimed_gap = row.get("gap_pct")
        if pd.notna(claimed_gap) and claimed_gap != "":
            diff = abs(actual_gap - float(claimed_gap))
            if diff > TOLERANCE_PCT:
                issues.append(f"gap off by {diff:.2f}pp")

        claimed_spy = row.get("spy_pct")
        if pd.notna(claimed_spy) and claimed_spy != "" and actual_spy is not None:
            diff = abs(actual_spy - float(claimed_spy))
            if diff > TOLERANCE_PCT:
                issues.append(f"SPY off by {diff:.2f}pp")

        result["status"] = "FLAG" if issues else "PASS"
        result["issue"] = "; ".join(issues) if issues else ""

    except Exception as e:
        result["status"] = "ERROR"
        result["issue"] = str(e)

    return result


def main():
    candidates = pd.read_csv(CANDIDATES_FILE)
    tickers = candidates["ticker"].unique()

    print(f"Downloading full-year history for {len(tickers)} tickers + SPY...")
    ticker_data = {t: get_full_history(t) for t in tickers}
    spy_df = get_full_history("SPY")

    print(f"Verifying {len(candidates)} rows...")
    results = []
    for i, row in candidates.iterrows():
        r = verify_row(row, spy_df, ticker_data[row["ticker"]])
        results.append(r)
        print(f"  [{r['status']}] {r['ticker']} {r['move_date']} (move={r.get('actual_move_pct')}, "
              f"gap={r.get('actual_gap_pct')})" + (f" -- {r['issue']}" if r.get("issue") else ""))

    report = pd.DataFrame(results)
    # put the key comparison columns up front for readability
    cols = ["ticker", "move_date", "direction", "timing",
            "claimed_move_pct", "actual_move_pct",
            "claimed_gap_pct", "actual_gap_pct",
            "claimed_spy_pct", "actual_spy_pct",
            "prev_trading_day", "status", "issue"]
    cols = [c for c in cols if c in report.columns]
    report = report[cols]
    report.to_csv(REPORT_FILE, index=False)

    n_pass = (report["status"] == "PASS").sum()
    n_flag = (report["status"] == "FLAG").sum()
    n_err = (report["status"] == "ERROR").sum()
    print(f"\nDone. PASS={n_pass}  FLAG={n_flag}  ERROR={n_err}")
    print(f"Full report written to {REPORT_FILE}")
    if n_flag or n_err:
        print("\nFlagged/errored rows are worth checking by hand first —")
        print("common real causes: adjusted vs. unadjusted close, wrong date/ticker,")
        print("or the original source using a slightly different close-time definition.")


if __name__ == "__main__":
    main()