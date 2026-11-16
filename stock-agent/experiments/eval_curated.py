"""Evaluate the agent against a HAND-CURATED benchmark (experiments/data/curated/curated_events.json).

You research and label the events yourself (real news that moved a stock, verified by hand - the AI
never helps curate). This script only TESTS the agent against your file, two ways, off the SAME rows:

  REASONING-ONLY  - we take the real forecast for that (ticker, date) but swap in YOUR curated pre-move
                    headlines as the news the agent reads. Tests purely: given the correct news, does the
                    agent reach the right verdict? (isolates reasoning from retrieval)

  FULL PIPELINE   - we run the real vet_forecast path: the agent fetches its OWN news (Finnhub/GDELT,
                    no-lookahead) and the real forecast, then judges. Tests the whole real-prompt flow.
                    We also measure RETRIEVAL RECALL: did the pipeline's own news even surface the
                    catalyst you found?

The gap between the two catch-rates is the punchline: if REASONING-ONLY catches an event but FULL
PIPELINE misses it, the failure was RETRIEVAL (our news source didn't find the story), not reasoning.

Ground truth = your labels. For 'event' rows the agent SHOULD flag caution/reject; for 'calm' rows it
SHOULD trust; 'unexplained' rows (big move, no news) test honesty - the agent should not invent a cause.

Run from the project root (fill the template first):
    ./.venv/Scripts/python.exe experiments/eval_curated.py
"""
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import tools, vetting  # noqa: E402

CURATED_DIR = ROOT / "experiments" / "data" / "curated"
DATA = CURATED_DIR / "curated_events.json"     # default; a .csv is auto-detected too (see _load_events)
OUT = ROOT / "experiments" / "results" / "curated_eval.txt"


def _load_events(path):
    """Return the list of event dicts from either a .json file or a Google-Sheets-style .csv.

    CSV columns (order does not matter - read by name): ticker, move_date, direction, label_type,
    catalyst, why, expected_verdict, headline_title, headline_url, headline_published, notes. Optional
    extra columns are kept when useful: 'status' (only rows == 'Done' are scored) and 'search_window'
    (used to split results overnight vs intraday). One row per HEADLINE - for an event with several
    headlines, repeat ticker+move_date on extra rows and fill only the headline columns. Any preamble
    / legend rows above the real header are skipped, as are rows whose ticker is blank or starts '#'."""
    if path.suffix.lower() == ".json":
        return [r for r in json.loads(path.read_text(encoding="utf-8")).get("events", [])
                if not r.get("_example")]

    # csv.reader (not DictReader) so we can locate the real header row under any preamble/legend rows
    # and correctly handle multi-line quoted cells (the 'why'/'notes' fields span lines).
    with path.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.reader(f))
    hi = next((i for i, r in enumerate(raw) if r and r[0].strip() == "ticker"), None)
    if hi is None:
        return []
    header = [h.strip() for h in raw[hi]]

    events, order = {}, []
    for cells in raw[hi + 1:]:
        r = dict(zip(header, cells))
        tk = (r.get("ticker") or "").strip()
        if not tk or tk.startswith("#"):
            continue
        status = (r.get("status") or "").strip().lower()
        if status and status != "done":      # skip Not-Started / in-progress rows
            continue
        key = (tk, (r.get("move_date") or "").strip())
        if key not in events:
            notes = (r.get("notes") or "").strip()
            m = re.search(r"SPY\s*([+-]?\d+(?:\.\d+)?)\s*%", notes)   # market move that day, from notes
            events[key] = {
                "ticker": tk, "move_date": key[1],
                "direction": (r.get("direction") or "").strip(),
                "label_type": (r.get("label_type") or "event").strip() or "event",
                "catalyst": (r.get("catalyst") or "").strip(),
                "why": (r.get("why") or "").strip(),
                "expected_verdict": (r.get("expected_verdict") or "").strip(),
                "notes": notes,
                "search_window": (r.get("search_window") or "").strip(),
                "spy_pct": float(m.group(1)) if m else None,
                "headlines": [],
            }
            order.append(key)
        url = (r.get("headline_url") or "").strip()
        title = (r.get("headline_title") or "").strip()
        if url or title:
            events[key]["headlines"].append({
                "title": title, "url": url,
                "published": (r.get("headline_published") or "").strip()})
    return [events[k] for k in order]


def _to_ts(published):
    """ISO timestamp -> unix seconds (UTC). Returns None if unparseable."""
    if not published:
        return None
    try:
        s = published.replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    except ValueError:
        return None


def _curated_news(row):
    """Convert the hand-entered headlines into the bundle's news_before shape."""
    out = []
    for h in row.get("headlines", []):
        out.append({"title": h.get("title"), "url": h.get("url"),
                    "seendate": h.get("published"), "ts": _to_ts(h.get("published")),
                    "source": "curated"})
    return out


def _norm_url(u):
    if not u:
        return ""
    u = u.lower().split("?")[0].rstrip("/")
    return re.sub(r"^https?://(www\.)?", "", u)


def _norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower())


def _retrieval_recall(curated, retrieved):
    """Fraction of curated headlines the pipeline's own news surfaced (URL match, or strong title
    overlap as a fallback for the same story reported by a different outlet)."""
    if not curated:
        return None
    r_urls = {_norm_url(n.get("url")) for n in retrieved}
    r_titles = [_norm_title(n.get("title")) for n in retrieved]
    hits = 0
    for c in curated:
        cu, ct = _norm_url(c.get("url")), _norm_title(c.get("title"))
        if cu and cu in r_urls:
            hits += 1
            continue
        ctoks = set(ct.split())
        if ctoks and any(len(ctoks & set(rt.split())) / len(ctoks) >= 0.6 for rt in r_titles):
            hits += 1
    return hits / len(curated)


def _verdict(bundle, mode):
    v, err = vetting.judge(bundle, mode=mode)
    if err or v is None:
        return None
    return v.stance


MODES = ("investor", "analyst")   # user-facing sensitivity modes (investor=precise, analyst=wide net)


def _pick_source(explicit=None):
    """Which file to read: an explicit path, else the .csv if present, else the .json default."""
    if explicit:
        return Path(explicit)
    csv_path = CURATED_DIR / "curated_events.csv"
    return csv_path if csv_path.exists() else DATA


def run(path=None):
    src = _pick_source(path)
    rows = _load_events(src)
    print(f"reading {src.name}")
    if not rows:
        print(f"No curated events yet in {src.name}. Fill it in (delete the example rows and add "
              f"your own - CSV rows starting with '#' are ignored), then re-run.")
        return

    recs = []
    for row in rows:
        tk, date, lt = row["ticker"], row["move_date"], row.get("label_type", "event")
        bundle = tools.run_tool("vet_forecast", {"ticker": tk, "as_of_date": date})
        if not bundle.get("ok"):
            print(f"[skip] {tk} {date}: {bundle.get('error')}")
            continue

        curated = _curated_news(row)
        reasoning_bundle = {**bundle, "news_before": curated} if curated else None
        recall = _retrieval_recall(curated, bundle.get("news_before", []))
        actual = bundle.get("actual", {})
        sw = (row.get("search_window") or "").lower()
        when = "intraday" if "trading hours" in sw else "overnight" if "evening before" in sw else "?"

        # company-specific vs macro: if the whole market moved >=1% that day, the stock's move is
        # largely beta, not a company catalyst the forecast "missed" - a muddier test case.
        spy = row.get("spy_pct")
        catalyst_type = "macro" if (spy is not None and abs(spy) >= 1.0) else "company"
        rec = {"ticker": tk, "date": date, "label_type": lt, "direction": row.get("direction"),
               "expected": row.get("expected_verdict"), "recall": recall, "when": when,
               "catalyst_type": catalyst_type, "spy_pct": spy,
               "realized": actual.get("realized_pct") if actual.get("available") else None}
        # both sensitivity modes, each in full-pipeline and (for events) reasoning-only
        for mode in MODES:
            rec[f"{mode}_full"] = _verdict(bundle, mode)
            rec[f"{mode}_reasoning"] = _verdict(reasoning_bundle, mode) if reasoning_bundle else None
        recs.append(rec)
        rc = "n/a" if recall is None else f"{recall*100:.0f}%"
        print(f"{tk:5} {date}  [{lt:11}] {when:9} "
              f"investor(r={str(rec['investor_reasoning']):7} p={str(rec['investor_full']):7}) "
              f"analyst(r={str(rec['analyst_reasoning']):7} p={str(rec['analyst_full']):7}) "
              f"recall={rc}")

    if not recs:
        print("no rows scored")
        return

    def flagged(stance):
        return stance in ("caution", "reject")

    events = [r for r in recs if r["label_type"] in ("event", "unexplained")]
    calm = [r for r in recs if r["label_type"] == "calm"]

    def rate(subset, key, want_flag):
        vals = [r[key] for r in subset if r[key] is not None]
        if not vals:
            return None
        return sum(flagged(s) == want_flag for s in vals) / len(vals)

    def pct(x):
        return "n/a " if x is None else f"{x*100:3.0f}%"

    L = []
    L.append(f"Hand-curated benchmark - {len(recs)} rows "
             f"({len(events)} event/unexplained, {len(calm)} calm)")
    L.append("Ground truth = your hand labels. Event -> agent should flag caution/reject; "
             "calm -> agent should trust. Two user-facing modes compared: INVESTOR (precise - only flag "
             "genuine threats; for long-term holders) vs ANALYST (wide net - flag any material "
             "catalyst; for active due-diligence / risk desks).")
    L.append("")
    L.append("=== SENSITIVITY TRADE-OFF (the headline table) ===")
    L.append(f"{'mode':11} {'catch (reasoning)':18} {'catch (pipeline)':17} {'calm trust':11} "
             f"{'discrimination':14}")
    for mode in MODES:
        cr = rate(events, f"{mode}_reasoning", True)
        cp = rate(events, f"{mode}_full", True)
        ct = rate(calm, f"{mode}_full", False)
        fe, fc = rate(events, f"{mode}_full", True), rate(calm, f"{mode}_full", True)
        disc = (fe - fc) if (fe is not None and fc is not None) else None
        L.append(f"{mode:11} {pct(cr):18} {pct(cp):17} {pct(ct):11} "
                 f"{(pct(disc) if disc is None else f'{disc*100:+3.0f} pts'):14}")
    L.append("")
    L.append("Read it: higher catch-rate = catches more real events (recall); higher calm-trust = "
             "fewer false alarms (precision). Analyst trades calm-trust for more catch vs Investor.")
    L.append("")

    # COMPANY-SPECIFIC vs MACRO: the fair test. On a company catalyst (earnings, deal) the price-only
    # model genuinely missed news, so the agent SHOULD flag. On a market-wide day (|SPY|>=1%) the move
    # is mostly beta, not company news, so the 'caution' label itself is debatable - a muddier case.
    company = [r for r in events if r["catalyst_type"] == "company"]
    macro = [r for r in events if r["catalyst_type"] == "macro"]
    L.append("=== FAIR TEST: company-specific vs market-wide events ===")
    L.append(f"(company = |SPY| < 1% that day; macro = |SPY| >= 1%, mostly market beta)")
    for label, grp in (("company-specific", company), ("market-wide (macro)", macro)):
        if not grp:
            continue
        L.append(f"  {label} (n={len(grp)}): "
                 f"investor catch reasoning {pct(rate(grp, 'investor_reasoning', True))} / "
                 f"pipeline {pct(rate(grp, 'investor_full', True))}  |  "
                 f"analyst catch reasoning {pct(rate(grp, 'analyst_reasoning', True))} / "
                 f"pipeline {pct(rate(grp, 'analyst_full', True))}")
    L.append("  (expect: catch-rate HIGHER on company-specific events - that is the case the use "
             "case is actually about. Macro days have debatable ground truth.)")
    L.append("")

    recalls = [r["recall"] for r in events if r["recall"] is not None]
    if recalls:
        L.append(f"RETRIEVAL RECALL (pipeline's own news surfaced your catalyst): "
                 f"{sum(recalls)/len(recalls)*100:.0f}% avg over {len(recalls)} events. Low recall caps "
                 f"pipeline catch-rate in BOTH modes - it can't flag a story it never fetched.")
        L.append("")

    # OVERNIGHT vs INTRADAY (using analyst pipeline, which flags most) - shows the open-cut effect
    for when in ("overnight", "intraday"):
        grp = [r for r in events if r["when"] == when]
        if not grp:
            continue
        rec_avg = [x for x in (r["recall"] for r in grp) if x is not None]
        rec_avg = sum(rec_avg) / len(rec_avg) if rec_avg else None
        L.append(f"  {when.upper()} events (n={len(grp)}): analyst catch "
                 f"reasoning {pct(rate(grp, 'analyst_reasoning', True))} / "
                 f"pipeline {pct(rate(grp, 'analyst_full', True))}, recall {pct(rec_avg)}")
    L.append("  (open-cut drops intraday news; see RRL-FINDINGS.md section 8.)")
    L.append("")
    L.append("Per-event  (I=investor, A=analyst; r=reasoning-only, p=full-pipeline):")
    for r in recs:
        rc = "n/a" if r["recall"] is None else f"{r['recall']*100:.0f}%"
        rz = "n/a" if r["realized"] is None else f"{r['realized']:+.1f}%"
        L.append(f"  {r['ticker']:5} {r['date']} [{r['label_type']:11}] {r['when']:9} "
                 f"exp={str(r['expected']):8} "
                 f"I(r={str(r['investor_reasoning']):7} p={str(r['investor_full']):7}) "
                 f"A(r={str(r['analyst_reasoning']):7} p={str(r['analyst_full']):7}) "
                 f"recall={rc:5} realized={rz}")
    report = "\n".join(L)
    print("\n" + report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
