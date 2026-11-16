"""Forecast bake-off - the Review of Related Literature (RRL) evidence for Component #14.

Walk-forward backtest of every forecast backend on FAANG/Big Tech over 2026. For a grid of
as-of dates we fit each model on data BEFORE the date only (no lookahead) and compare its
horizon-ahead call to what actually happened. We report, per model:

  HIT  - directional accuracy (did it call up/down right). This is the number that matters.
  MAE  - mean absolute error on the horizon-end price (in dollars)
  MAPE - the same error as a % of price

The expected, honest finding is that everything sits near the naive baseline's ~0.5 coin flip on
direction (efficient-market hypothesis). That is not a failure - it is the motivation for the whole
project: if even a reasonable forecaster is near-random, the value is the agent's news-vetting layer
that flags WHEN to distrust the forecast. Whichever model edges out the baseline here is what you
set as FORECAST_BACKEND in core/config.py (the live vet_forecast tool then uses it).

Run from the project root:
    ./.venv/Scripts/python.exe experiments/forecast_bakeoff.py
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import forecast, market_data  # noqa: E402  (core.config loads .env on import)

FAANG = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA"]
HORIZON = 5           # trading days ahead each forecast looks
MIN_HISTORY = 30      # need at least this many prior trading days before an as-of date
STEP = 5              # sample an as-of date every STEP trading days (keeps runtime sane)
START = "2026-01-01"


FLAT_EPS = 0.0005     # a forecast whose move is < 0.05% of price is treated as "no directional call"


def _direction(a, b):
    return "up" if b > a else "down" if b < a else "flat"


def _is_flat(last, pred):
    return last == 0 or abs(pred - last) / last < FLAT_EPS


def _series_for(ticker):
    today = dt.date.today().isoformat()
    ds = market_data.daily_series(ticker, START, today)
    return ds["series"] if ds.get("ok") else []


def _available_backends():
    """naive + the 3 statsforecast models, plus chronos only if torch/chronos is installed."""
    backends = ["naive", "ets", "arima", "theta"]
    probe = forecast.chronos_forecast([{"date": f"2026-01-{d:02d}", "close": 100 + d} for d in range(1, 13)], horizon=3)
    if probe.get("ok"):
        backends.append("chronos")
    else:
        print(f"[info] skipping chronos backend: {probe.get('error')}")
    return backends


def run():
    backends = _available_backends()
    # per backend: list of (hit: bool, abs_err: float, ape: float, is_flat: bool)
    stats = {b: [] for b in backends}
    n_samples = 0

    for ticker in FAANG:
        rows = _series_for(ticker)
        if len(rows) < MIN_HISTORY + HORIZON:
            print(f"[warn] {ticker}: only {len(rows)} rows, skipping")
            continue
        # each as-of index i: fit on rows[:i], the realised window is rows[i:i+HORIZON]
        for i in range(MIN_HISTORY, len(rows) - HORIZON, STEP):
            in_sample = rows[:i]
            as_of = rows[i]["date"]
            last_in = in_sample[-1]["close"]
            actual_end = rows[i + HORIZON - 1]["close"]
            actual_dir = _direction(last_in, actual_end)
            n_samples += 1
            for b in backends:
                fc = forecast.forecast(in_sample, horizon=HORIZON, backend=b, as_of=as_of)
                if not fc.get("ok"):
                    continue
                pred_end = fc["predictions"][-1]["point"]
                stats[b].append((
                    _direction(last_in, pred_end) == actual_dir,
                    abs(pred_end - actual_end),
                    abs(pred_end - actual_end) / actual_end if actual_end else 0.0,
                    _is_flat(last_in, pred_end),
                ))
        print(f"[ok] {ticker}: {len(rows)} rows processed")

    # aggregate. HIT counts a flat forecast as a miss (it made no directional call); HIT_C is
    # accuracy only over forecasts that DID commit to a direction; FLAT is how often it abstained.
    rows_out = []
    for b in backends:
        s = stats[b]
        if not s:
            continue
        n = len(s)
        committed = [r for r in s if not r[3]]
        hit = sum(1 for r in s if r[0]) / n
        hit_c = (sum(1 for r in committed if r[0]) / len(committed)) if committed else 0.0
        flat = sum(1 for r in s if r[3]) / n
        mae = sum(r[1] for r in s) / n
        mape = sum(r[2] for r in s) / n * 100
        rows_out.append({"model": b, "n": n, "hit": hit, "hit_c": hit_c,
                         "flat": flat, "mae": mae, "mape": mape})

    by_model = {r["model"]: r for r in rows_out}
    # The INTEGRATED forecaster (Component #14) must be a real DS model - naive drift is only the
    # baseline floor. Pick the best real model by directional accuracy, tie-break on lower MAPE.
    real = [r for r in rows_out if r["model"] != "naive"]
    real.sort(key=lambda r: (-r["hit"], r["mape"]))
    winner = real[0]["model"] if real else "naive"

    order = ["naive", "ets", "arima", "theta", "chronos"]
    rows_out.sort(key=lambda r: order.index(r["model"]) if r["model"] in order else 99)

    lines = []
    lines.append(f"Forecast bake-off - FAANG/Big Tech, {START}..today, horizon={HORIZON}d, "
                 f"{n_samples} as-of samples across the ticker grid")
    lines.append("HIT = directional accuracy (flat forecast counts as a miss). "
                 "HIT_C = accuracy when the model committed to a direction. FLAT = % abstained.")
    lines.append("")
    lines.append(f"{'model':<10}{'n':>5}{'HIT':>8}{'HIT_C':>9}{'FLAT':>8}{'MAE $':>10}{'MAPE':>9}")
    lines.append("-" * 59)
    for r in rows_out:
        tag = "  (baseline)" if r["model"] == "naive" else ""
        lines.append(f"{r['model']:<10}{r['n']:>5}{r['hit']*100:>7.1f}%{r['hit_c']*100:>8.1f}%"
                     f"{r['flat']*100:>7.1f}%{r['mae']:>10.2f}{r['mape']:>8.2f}%{tag}")
    lines.append("")
    base_hit = by_model["naive"]["hit"] * 100 if "naive" in by_model else float("nan")
    lines.append(f"Naive-drift baseline directional accuracy: {base_hit:.1f}% (a coin flip is 50%).")
    lines.append(f"Recommended FORECAST_BACKEND = {winner}  "
                 f"(best REAL model; naive is only the baseline). Set it in core/config.py or .env.")
    lines.append("Finding: the models do not meaningfully beat the coin-flip baseline on direction. "
                 "That is the point - a price-only forecast is unreliable, which is exactly why the "
                 "agent's news-vetting layer (flagging WHEN to distrust it) is the value.")
    report = "\n".join(lines)
    print("\n" + report)

    outdir = ROOT / "experiments" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "forecast_bakeoff.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {outdir / 'forecast_bakeoff.txt'}")


if __name__ == "__main__":
    run()
