"""Discrimination, measured against REAL event labels (EDT Event_detection).

EDT's Event_detection files are token-level annotated news: each token is tagged with a corporate
event type (Acquisition, Guidance Increase, Dividend Cut, Special Dividend, Stock Split, ...) or
`O` (no event). We reconstruct documents and label each one:
  - EVENT = contains at least one annotated corporate event (a real, human-labeled catalyst)
  - CALM  = all tokens are `O` (no material event)

Then we ask the agent, reading only the article text, whether a material stock-moving event is
present. Discrimination = how much more often it flags EVENT docs than CALM docs. This is the
honest, dataset-backed discrimination metric (real labels, independent of price) and it measures
the core beyond-midterm skill: reading news to find a real catalyst.

The script computes all metrics itself and writes them to experiments/results/edt_events.txt.

Run from the project root (optionally pass how many docs PER CLASS to test, default 75):
    ./.venv/Scripts/python.exe experiments/eval_edt_events.py 75
"""
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm  # noqa: E402

EDT = ROOT / "experiments" / "data" / "edt" / "Event_detection" / "dev.txt"
OUT = ROOT / "experiments" / "results" / "edt_events.txt"
MIN_TOKENS = 20
SEED = 42

TYPE = {"A": "Acquisition", "CT": "Clinical Trial", "RD": "Regular Dividend", "DC": "Dividend Cut",
        "DI": "Dividend Increase", "GI": "Guidance Increase", "GC": "Guidance Change",
        "NC": "New Contract", "RSS": "Reverse Stock Split", "SD": "Special Dividend",
        "SR": "Stock Repurchase", "SS": "Stock Split"}


def _load_docs(path):
    """Parse the BIO file into documents: (text, set_of_event_types)."""
    docs, toks, types = [], [], set()
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if len(toks) >= MIN_TOKENS:
                docs.append((" ".join(toks), types))
            toks, types = [], set()
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        tok, lab = parts
        toks.append(tok)
        if lab != "O":
            types.add(lab.split("-")[-1])
    if len(toks) >= MIN_TOKENS:
        docs.append((" ".join(toks), types))
    return docs


def _ask_has_event(text):
    prompt = (
        "You are a financial news analyst. Does the following news report a MATERIAL corporate event "
        "likely to move the company's stock - such as an acquisition, earnings guidance change, "
        "dividend change, stock split, major new contract, clinical-trial result, or share buyback? "
        "Answer on the first line with ONLY 'YES' or 'NO'.\n\nNEWS:\n" + text[:1600]
    )
    try:
        r = llm.complete([{"role": "user", "content": prompt}], temperature=0.0).strip().upper()
        return "YES" if re.search(r"\bYES\b", r) else "NO"
    except Exception:
        return "ERR"


def run(n_per_class=75):
    if not EDT.exists():
        print(f"missing {EDT}")
        return
    docs = _load_docs(EDT)
    events = [(t, ty) for t, ty in docs if ty]
    calm = [(t, ty) for t, ty in docs if not ty]

    rng = random.Random(SEED)
    rng.shuffle(events)
    rng.shuffle(calm)
    ev = events[:n_per_class]
    ca = calm[:n_per_class]

    lines = [f"EDT event-detection discrimination (real gold labels, dev.txt)",
             f"parsed {len(docs)} docs: {len(events)} event / {len(calm)} calm; "
             f"testing {len(ev)} event + {len(ca)} calm (seed {SEED})", ""]

    ev_flag = ca_flag = 0
    misses, false_alarms = [], []
    for text, types in ev:
        a = _ask_has_event(text)
        if a == "YES":
            ev_flag += 1
        else:
            misses.append("/".join(sorted(TYPE.get(t, t) for t in types)))
    for text, _ in ca:
        a = _ask_has_event(text)
        if a == "YES":
            ca_flag += 1
            false_alarms.append(text[:70])

    ne, nc = len(ev), len(ca)
    ev_rate, ca_rate = ev_flag / ne, ca_flag / nc
    # precision/recall treating "flag" as positive, EVENT as the true-positive class
    tp, fn, fp, tn = ev_flag, ne - ev_flag, ca_flag, nc - ca_flag
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / (ne + nc)

    lines += [
        "=== RESULTS (computed by this script) ===",
        f"recall on real catalysts (flags EVENT docs): {ev_rate*100:.0f}%  ({tp}/{ne})",
        f"false-alarm rate (flags CALM docs):          {ca_rate*100:.0f}%  ({fp}/{nc})",
        f"DISCRIMINATION gap (event - calm):           {(ev_rate-ca_rate)*100:+.0f} pts",
        f"precision {prec*100:.0f}%  recall {rec*100:.0f}%  F1 {f1*100:.0f}%  accuracy {acc*100:.0f}%",
        f"(n={ne+nc}, chance = 50%)",
        "",
        f"missed events ({len(misses)}): " + (", ".join(misses) if misses else "none"),
        f"false alarms on calm ({len(false_alarms)}):",
    ] + [f"   - {t}" for t in false_alarms]

    report = "\n".join(lines)
    print(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    run(n)
