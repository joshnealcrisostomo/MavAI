"""'Consistent-then-blindsided' detector - the professor's showcase scenario.

Find events where the price-only forecaster was directionally CORRECT for several days running, then
got the event day WRONG (blindsided by news it never saw) - and the vetting agent flagged it. That
single sequence is the value proposition in one slide: the algorithm looks reliable, until news it
can't see breaks and the agent is the one that catches it.

Honesty note for the write-up: a short correct streak on daily direction is well within chance (a
coin flip clears 4-5 in a row a few % of the time). This is an ILLUSTRATIVE example, not evidence the
forecaster is genuinely predictive - ~7 months of data cannot support that stronger claim.

Run from the project root:
    ./.venv/Scripts/python.exe experiments/blindsided.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core import forecast, market_data, tools, vetting  # noqa: E402
import events  # noqa: E402

START = "2026-01-01"
LOOKBACK = 8      # trading days of streak to check before the event
MIN_HISTORY = 20  # need this many closes before the first streak day to fit a forecast


def _dir(a, b):
    return "up" if b > a else "down" if b < a else "flat"


def _one_day_dir(in_sample):
    """Theta (the shipped backend) 1-day-ahead directional call from an in-sample series."""
    fc = forecast.forecast(in_sample, horizon=1, backend="auto")
    if not fc.get("ok"):
        return None
    return _dir(fc["last_close"], fc["predictions"][-1]["point"])


def rolling_streak(ticker, event_date, lookback=LOOKBACK):
    """Consecutive correct 1-day-ahead calls in the `lookback` days before the event, then whether
    the event-day call was wrong."""
    import datetime as dt
    # fetch a bit past the event so the event day itself is in the series
    end = (dt.date.fromisoformat(event_date) + dt.timedelta(days=6)).isoformat()
    ds = market_data.daily_series(ticker, START, end)
    if not ds.get("ok"):
        return None
    rows = ds["series"]
    idx = next((i for i, r in enumerate(rows) if r["date"] == event_date), None)
    if idx is None or idx < MIN_HISTORY + 1:
        return None

    # streak over the days strictly before the event
    streak, streak_dates = 0, []
    for i in range(idx - 1, max(MIN_HISTORY, idx - lookback) - 1, -1):
        pred = _one_day_dir(rows[:i])
        actual = _dir(rows[i - 1]["close"], rows[i]["close"])
        if pred is not None and pred == actual and pred != "flat":
            streak += 1
            streak_dates.append(rows[i]["date"])
        else:
            break

    # the event day itself
    event_pred = _one_day_dir(rows[:idx])
    event_actual = _dir(rows[idx - 1]["close"], rows[idx]["close"])
    blindsided = event_pred is not None and event_pred != event_actual

    return {
        "ticker": ticker, "event_date": event_date,
        "streak_len": streak, "streak_dates": list(reversed(streak_dates)),
        "event_pred_dir": event_pred, "event_actual_dir": event_actual,
        "blindsided": blindsided,
    }


def run():
    cands = events.auto_events(big_per=3, calm_per=0)   # streaks are only interesting around big moves
    results = []
    for e in cands:
        r = rolling_streak(e["ticker"], e["event_date"])
        if r:
            results.append(r)
            print(f"{r['ticker']:5} {r['event_date']}  streak={r['streak_len']}  "
                  f"event pred={r['event_pred_dir']} actual={r['event_actual_dir']}  "
                  f"blindsided={r['blindsided']}")

    shows = [r for r in results if r["blindsided"] and r["streak_len"] >= 2]
    # Judge each candidate so we can prefer the ones the agent actually CAUGHT (flagged caution/
    # reject). "Model was right for days, got blindsided, and the agent caught it" is the showcase;
    # "model blindsided and the agent also trusted it" is not.
    for r in shows:
        b = tools.run_tool("vet_forecast", {"ticker": r["ticker"], "as_of_date": r["event_date"]})
        v, _ = vetting.judge(b)
        r["agent_stance"] = v.stance if v else "?"
        r["agent_conf"] = v.confidence if v else "?"
        r["agent_cited"] = len(v.cited_news) if v else 0
        r["agent_caught"] = bool(v and v.stance in ("caution", "reject"))
    shows.sort(key=lambda r: (r["agent_caught"], r["streak_len"]), reverse=True)

    lines = ["'Consistent-then-blindsided' scan - FAANG 2026 (theta 1-day-ahead)", ""]
    if not shows:
        lines.append("No event with a >=2-day correct streak that then flipped. (Expected sometimes: "
                     "the forecaster is near-random, so clean streaks are down to chance.)")
    else:
        best = shows[0]
        caught = best["agent_caught"]
        lines.append(f"BEST SHOWCASE: {best['ticker']} on {best['event_date']}"
                     + ("  <- agent CAUGHT it" if caught else "  (agent did NOT catch this one)"))
        lines.append(f"  correct 1-day-ahead calls for {best['streak_len']} days running "
                     f"({', '.join(best['streak_dates'])}),")
        lines.append(f"  then predicted {best['event_pred_dir']} for {best['event_date']} but the "
                     f"stock actually went {best['event_actual_dir']} - blindsided.")
        lines.append(f"  vetting agent's call that day: stance={best['agent_stance']}, "
                     f"confidence={best['agent_conf']} (cited {best['agent_cited']} pre-event headlines)")
        lines.append("")
        lines.append("All blindsided-after-streak events (agent_stance):")
        for r in shows:
            lines.append(f"  {r['ticker']} {r['event_date']}  streak {r['streak_len']}  "
                         f"agent={r['agent_stance']}" + ("  [caught]" if r["agent_caught"] else ""))
    lines.append("")
    lines.append("Caveat: a short daily-direction streak is within chance; this is an illustrative "
                 "example of the value proposition, not proof the forecaster is predictive.")
    report = "\n".join(lines)
    print("\n" + report)

    outdir = ROOT / "experiments" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "blindsided.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {outdir / 'blindsided.txt'}")


if __name__ == "__main__":
    run()
