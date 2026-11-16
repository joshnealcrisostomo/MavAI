"""FAANG-2026 catalyst-detection check using SEC 8-K filings as the ground truth.

There is no labeled event dataset for 2026 (academic sets lag by years), so we use an OFFICIAL,
free basis instead: an 8-K is the SEC form a company files to disclose a MATERIAL event (earnings,
acquisitions, executive changes, major agreements). So:
  - EVENT day = the company filed an 8-K on/around that date
  - CALM day  = a trading day with no 8-K nearby
For each day we fetch that day's news (Finnhub) and ask the agent whether a material event is
present, then measure discrimination the same way as the EDT eval - but now on our actual scope
(FAANG, 2026). All metrics are computed by this script.

Caveat: an 8-K marks a material FILING; the day's news headlines don't always feature it (e.g. a
quiet officer-change filing), so this is a noisier ground truth than EDT's annotated news. It's an
on-scope confirmation, not a replacement for the EDT number.

Run from the project root:
    ./.venv/Scripts/python.exe experiments/eval_faang_8k.py 8
"""
import datetime as dt
import random
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm, news_hist  # noqa: E402

OUT = ROOT / "experiments" / "results" / "faang_8k.txt"
UA = {"User-Agent": "STAI100 Capstone research paolo@octane.ai"}
SEED = 42

CIK = {"AAPL": 320193, "MSFT": 789019, "AMZN": 1018724, "GOOGL": 1652044,
       "META": 1326801, "NVDA": 1045810, "TSLA": 1318605}


def _eight_k_dates(cik):
    """2026 8-K filing dates for a CIK, from SEC EDGAR's submissions feed."""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    try:
        rec = requests.get(url, headers=UA, timeout=20).json()["filings"]["recent"]
    except Exception:
        return []
    return sorted({d for f, d in zip(rec["form"], rec["filingDate"])
                   if f == "8-K" and d.startswith("2026")})


def _calm_dates(eight_ks, k):
    """k weekday dates in 2026 that are NOT within 5 days of any 8-K for this ticker."""
    ek = {dt.date.fromisoformat(d) for d in eight_ks}
    cutoff = dt.date.today() - dt.timedelta(days=10)   # leave room for a news window
    day = dt.date(2026, 2, 1)
    cands = []
    while day < cutoff:
        if day.weekday() < 5 and all(abs((day - e).days) > 5 for e in ek):
            cands.append(day.isoformat())
        day += dt.timedelta(days=1)
    rng = random.Random(SEED + hash(str(eight_ks)) % 1000)
    rng.shuffle(cands)
    return cands[:k]


def _news_text(ticker, date):
    """Headlines around a date (day-before .. day-of), joined for the agent to read."""
    d = dt.date.fromisoformat(date)
    items = news_hist.news_for_ticker(ticker, (d - dt.timedelta(days=1)).isoformat(), date, 60)
    titles = [n["title"] for n in items if n.get("title")][:12]
    return " | ".join(titles)


def _ask_material(ticker, text):
    prompt = (
        f"You are a financial news analyst. Based on these {ticker} news headlines, is there a "
        "MATERIAL corporate event likely to move the stock (acquisition, earnings/guidance change, "
        "executive change, major agreement, dividend/buyback, product/regulatory/legal event)? "
        "Answer on the first line with ONLY 'YES' or 'NO'.\n\nHEADLINES:\n" + text[:1600]
    )
    try:
        r = llm.complete([{"role": "user", "content": prompt}], temperature=0.0).strip().upper()
        return "YES" if re.search(r"\bYES\b", r) else "NO"
    except Exception:
        return "ERR"


def run(n_per_class=8):
    event_pool, calm_pool = [], []
    for tk, cik in CIK.items():
        eks = _eight_k_dates(cik)
        event_pool += [(tk, d) for d in eks]
        calm_pool += [(tk, d) for d in _calm_dates(eks, 3)]
    rng = random.Random(SEED)
    rng.shuffle(event_pool)
    rng.shuffle(calm_pool)

    # only keep days that actually have news to read (else the test is trivial)
    def take(pool, n):
        out = []
        for tk, d in pool:
            txt = _news_text(tk, d)
            if txt:
                out.append((tk, d, txt))
            if len(out) >= n:
                break
        return out

    ev = take(event_pool, n_per_class)
    ca = take(calm_pool, n_per_class)

    lines = ["FAANG-2026 catalyst detection vs SEC 8-K ground truth",
             f"8-K days found across FAANG in 2026: {len(event_pool)}; "
             f"testing {len(ev)} event + {len(ca)} calm days (seed {SEED})", ""]

    ev_flag = ca_flag = 0
    for tk, d, txt in ev:
        a = _ask_material(tk, txt)
        ev_flag += (a == "YES")
        lines.append(f"  EVENT {tk} {d} (8-K)  agent={a}")
    for tk, d, txt in ca:
        a = _ask_material(tk, txt)
        ca_flag += (a == "YES")
        lines.append(f"  CALM  {tk} {d}        agent={a}")

    ne, nc = len(ev), len(ca)
    if ne and nc:
        ev_rate, ca_rate = ev_flag / ne, ca_flag / nc
        acc = (ev_flag + (nc - ca_flag)) / (ne + nc)
        lines += ["",
                  "=== RESULTS (computed by this script) ===",
                  f"flags EVENT (8-K) days: {ev_rate*100:.0f}%  ({ev_flag}/{ne})",
                  f"flags CALM days:        {ca_rate*100:.0f}%  ({ca_flag}/{nc})",
                  f"DISCRIMINATION gap:     {(ev_rate-ca_rate)*100:+.0f} pts",
                  f"accuracy: {acc*100:.0f}%  (n={ne+nc}, chance 50%)",
                  "Caveat: 8-K = material FILING; the day's headlines don't always feature it, so "
                  "this is a noisier, on-scope confirmation of the EDT capability number."]
    else:
        lines.append("not enough days with news; try a different range")

    report = "\n".join(lines)
    print(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
