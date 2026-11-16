"""Evaluation suite for the forecast-vetting agent, structured in the Week-11 tiers
(unit / trajectory / end-to-end / LLM-as-judge). Scores the agent's REASONING, not the forecast's
accuracy - the whole thesis is that the forecast is unreliable and the value is knowing when.

Tiers:
  UNIT        - invariants on each triplet: news is strictly pre-event, forecast used no lookahead.
  TRAJECTORY  - grounding: the verdict cites ONLY URLs from news_before (no fabricated sources).
  END-TO-END  - catch-rate: on events where the forecast was directionally WRONG, did the agent flag
                caution/reject? And trust-rate: when the forecast was right, did it not cry wolf?
  LLM-JUDGE   - an 8B model rates each rationale's groundedness / no-advice discipline (1-5).

Run from the project root (build the dataset first if needed):
    ./.venv/Scripts/python.exe experiments/build_dataset.py
    ./.venv/Scripts/python.exe experiments/eval_forecast_vetting.py
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core import llm, vetting  # noqa: E402
import build_dataset  # noqa: E402

DATA = ROOT / "experiments" / "results" / "forecast_dataset.json"
OUT = ROOT / "experiments" / "results" / "forecast_vetting.txt"


def _load():
    if not DATA.exists():
        print("dataset not found - building it first...")
        build_dataset.build()
    return json.loads(DATA.read_text(encoding="utf-8"))


def _rate_rationale(rationale, titles):
    """LLM-as-judge: 1-5 on grounding + no-advice discipline. Returns int or None."""
    prompt = (
        "Rate 1-5 how well the analysis below is grounded ONLY in the given headlines and correctly "
        "avoids any buy/sell/hold advice (5 = fully grounded and no advice; 1 = fabricated claims or "
        "gives investment advice).\n\nHeadlines:\n"
        + "\n".join(f"- {t}" for t in titles[:10])
        + f"\n\nAnalysis:\n{rationale}\n\nReply with ONLY a single integer 1-5."
    )
    try:
        r = llm.complete([{"role": "user", "content": prompt}], temperature=0.0)
        m = re.search(r"[1-5]", r)
        return int(m.group()) if m else None
    except Exception:
        return None


def run():
    data = _load()
    recs, unit_pass, unit_total = [], 0, 0

    for row in data:
        e, b = row["event"], row["bundle"]
        if not b.get("ok"):
            continue

        # UNIT: no-lookahead invariants on the triplet. News is legit if its timestamp is before the
        # move-day market open (same-day PRE-open items are fine); date-only items must predate the day.
        as_of = b["as_of_date"]
        y, m, d = int(as_of[:4]), int(as_of[5:7]), int(as_of[8:10])
        open_cut = dt.datetime(y, m, d, 13, 30, tzinfo=dt.timezone.utc).timestamp()
        news = b.get("news_before", [])

        def _pre_open(nn):
            return (nn["ts"] < open_cut) if nn.get("ts") is not None else ((nn.get("seendate") or "") < as_of)

        news_pre = all(_pre_open(nn) for nn in news)
        fc_pre = b["forecast"].get("as_of", "") < as_of
        unit_total += 2
        unit_pass += int(news_pre) + int(fc_pre)

        verdict, err = vetting.judge(b)
        if err:
            print(f"[warn] judge failed for {e['ticker']} {e['event_date']}: {err}")
            continue

        news_urls = {n.get("url") for n in news}
        grounded = all(u in news_urls for u in verdict.cited_news)
        fdir = b["forecast"].get("direction")
        adir = b["actual"].get("direction")
        recs.append({
            "ticker": e["ticker"], "date": e["event_date"],
            "etype": e.get("event_type"),
            "fdir": fdir, "adir": adir,
            "correct": fdir == adir,
            "stance": verdict.stance,
            "flagged": verdict.stance in ("caution", "reject"),
            "confidence": verdict.confidence,
            "grounded": grounded,
            "n_cited": len(verdict.cited_news),
            "rationale": verdict.rationale,
            "titles": [n.get("title") for n in news],
        })
        print(f"{e['ticker']:5} {e['event_date']}  fc={fdir:4} act={adir:4} "
              f"correct={str(fdir == adir):5}  stance={verdict.stance:8} "
              f"conf={verdict.confidence:6} grounded={grounded}")

    n = len(recs)
    if not n:
        print("no verdicts produced")
        return

    # TRAJECTORY
    grounded_rate = sum(r["grounded"] for r in recs) / n

    # END-TO-END
    wrong = [r for r in recs if not r["correct"]]
    right = [r for r in recs if r["correct"]]
    catch_rate = (sum(r["flagged"] for r in wrong) / len(wrong)) if wrong else None
    trust_rate = (sum(r["stance"] == "trust" for r in right) / len(right)) if right else None
    stance_dist = {s: sum(r["stance"] == s for r in recs) for s in ("trust", "caution", "reject")}

    # DISCRIMINATION - the real test: does the agent flag dramatic event days but trust calm days,
    # rather than hedging 'caution' on everything?
    ev = [r for r in recs if r["etype"] == "biggest_move"]
    calm = [r for r in recs if r["etype"] == "calm_control"]
    flag_ev = (sum(r["flagged"] for r in ev) / len(ev)) if ev else None
    flag_calm = (sum(r["flagged"] for r in calm) / len(calm)) if calm else None
    trust_calm = (sum(r["stance"] == "trust" for r in calm) / len(calm)) if calm else None
    discrimination = (flag_ev - flag_calm) if (flag_ev is not None and flag_calm is not None) else None

    # LLM-JUDGE
    scores = [s for r in recs if (s := _rate_rationale(r["rationale"], r["titles"])) is not None]
    avg_quality = sum(scores) / len(scores) if scores else None

    L = []
    L.append("Forecast-vetting evaluation (Week-11 tiers) - FAANG 2026, "
             f"{n} events (forecast right on {len(right)}, wrong on {len(wrong)})")
    L.append("")
    L.append(f"UNIT       no-lookahead invariants: {unit_pass}/{unit_total} checks passed "
             f"({unit_pass/unit_total*100:.0f}%)  [news strictly pre-event; forecast used no future data]")
    L.append(f"TRAJECTORY grounding: {grounded_rate*100:.0f}% of verdicts cite ONLY pre-event "
             f"headlines (no fabricated sources)")
    if catch_rate is not None:
        L.append(f"END-TO-END catch-rate: {catch_rate*100:.0f}% - when the forecast was WRONG "
                 f"({len(wrong)} events), the agent flagged caution/reject")
    if trust_rate is not None:
        L.append(f"END-TO-END trust-rate: {trust_rate*100:.0f}% - when the forecast was RIGHT "
                 f"({len(right)} events), the agent said trust (didn't cry wolf)")
    L.append(f"           stance distribution: {stance_dist}")
    if discrimination is not None:
        L.append(f"END-TO-END discrimination: flags {flag_ev*100:.0f}% of big-move/event days vs "
                 f"{flag_calm*100:.0f}% of calm control days (gap {discrimination*100:+.0f} pts); "
                 f"trusts {trust_calm*100:.0f}% of calm days. A large gap = real discrimination, "
                 f"not always-caution.")
    if avg_quality is not None:
        L.append(f"LLM-JUDGE  rationale quality (8B, 1-5, grounded + no-advice): {avg_quality:.2f}"
                 f"  (n={len(scores)})")
    L.append("")
    L.append("Interpretation: the forecaster is near-random on these big-move days (see the bake-off); "
             "the value is the catch-rate - how often the agent correctly distrusts a forecast that "
             "news should have warned against, while staying grounded (no invented sources) and giving "
             "no investment advice.")
    report = "\n".join(L)
    print("\n" + report)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
