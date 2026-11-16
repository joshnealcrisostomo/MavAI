"""Dataset-backed sanity check on the EDT ("Trade the Event") benchmark.

EDT gives real news articles + the real subsequent price move (1/2/3-day). It does NOT give a price
history to forecast from, so this is not the full forecast-vetting loop - it tests the CORE skill
underneath it: given a real news article, does the agent read the likely near-term direction
correctly? Ground truth is EDT's actual 1-day move (external, labeled). We filter to clear moves
(|1-day| >= threshold) so there's a real signal to get right.

Reads a slice via streaming (the file is ~1.6GB, 303k records - we never load it all).
Run from the project root:
    ./.venv/Scripts/python.exe experiments/eval_edt.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm  # noqa: E402

EDT = ROOT / "experiments" / "data" / "edt" / "Trading_benchmark" / "evaluate_news.json"
SCAN = 6000          # how many records to stream through
MOVE_THRESHOLD = 3.0  # keep only news whose actual 1-day move is at least this % (a real move)
N = 30                # how many big-move items to test the agent on


def _stream(path, limit):
    import ijson
    with open(path, encoding="utf-8") as f:
        for i, obj in enumerate(ijson.items(f, "item")):
            if i >= limit:
                break
            yield obj


def _move_pct(lab):
    try:
        s, e = float(lab["start_price_close"]), float(lab["end_price_1day"])
        return (e - s) / s * 100 if s else None
    except (TypeError, ValueError, KeyError):
        return None


def _ask_direction(ticker, text):
    prompt = (
        f"You are a financial news analyst. Based ONLY on the following news about {ticker}, is the "
        "stock likely to move UP or DOWN over the next 1-3 trading days? If the news has no clear "
        "directional implication, answer UNCLEAR. Reply with ONE word: UP, DOWN, or UNCLEAR.\n\n"
        f"NEWS:\n{text[:1500]}"
    )
    try:
        r = llm.complete([{"role": "user", "content": prompt}], temperature=0.0).strip().upper()
        m = re.search(r"\b(UP|DOWN|UNCLEAR)\b", r)
        return m.group(1) if m else "UNCLEAR"
    except Exception:
        return "ERR"


def run():
    if not EDT.exists():
        print(f"missing {EDT} - download the EDT Trading_benchmark first")
        return
    cands = []
    for obj in _stream(EDT, SCAN):
        mp = _move_pct(obj.get("labels", {}))
        if mp is None or abs(mp) < MOVE_THRESHOLD:
            continue
        text = f"{obj.get('title', '')}. {obj.get('text', '')}"
        cands.append((obj["labels"].get("ticker"), text, mp))
        if len(cands) >= N:
            break

    print(f"{len(cands)} EDT news items with a clear 1-day move (>= {MOVE_THRESHOLD}%)")
    hits = commit = 0
    for tk, text, mp in cands:
        pred = _ask_direction(tk, text)
        actual = "UP" if mp > 0 else "DOWN"
        if pred in ("UP", "DOWN"):
            commit += 1
            hits += (pred == actual)
        print(f"  {str(tk):6} actual={actual:4} ({mp:+.1f}%)  agent={pred}")

    print()
    if commit:
        print(f"Committed on {commit}/{len(cands)} (abstained {len(cands)-commit}). "
              f"Directional accuracy when it committed: {hits}/{commit} = {hits/commit*100:.0f}% "
              f"(50% = coin flip).")
    else:
        print("agent abstained on everything")
    print("Note: EDT news is often the catalyst itself (PR-wire), so this tests reading a real "
          "article's directional implication against the real market reaction - a dataset-backed "
          "check of the agent's news comprehension, not the full forecast-vetting loop.")


if __name__ == "__main__":
    run()
