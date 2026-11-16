"""FinBERT vs 8B-LLM sentiment on a HELD-OUT set (FiQA), to check if FinBERT's PhraseBank edge is real.

FinBERT was fine-tuned on Financial PhraseBank, so its 98% there (see eval_finbert.py) is in-domain
and may be memorised style. FiQA is a different financial-sentiment dataset FinBERT was NOT trained
on (mixed headlines + posts, continuous scores), so it's a fair out-of-domain test. We bucket FiQA's
continuous score into positive/negative/neutral to compare against FinBERT's 3-class output, and run
the same 8B LLM baseline as the in-domain eval.

Three honest outcomes, all fine for the write-up:
  - FinBERT still clearly wins  -> the edge generalises; keep it as the live path.
  - gap shrinks / LLM wins       -> the 98% was mostly PhraseBank-specific; keep FinBERT for
                                    determinism + being a trained DS model, but say so plainly.
  - both close                   -> sentiment at this granularity is easy; the value-add is elsewhere.

Run from the project root (optional sample size):
    ./.venv/Scripts/python.exe experiments/eval_finbert_heldout.py 150
"""
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, llm, sentiment  # noqa: E402

OUT = ROOT / "experiments" / "results" / "finbert_heldout.txt"
SEED = 42
LABELS = ["negative", "neutral", "positive"]
POS_T, NEG_T = 0.1, -0.1   # FiQA score -> class thresholds


def _load_fiqa():
    from datasets import load_dataset
    ds = load_dataset("TheFinAI/fiqa-sentiment-classification", "default", split="train")
    rows = []
    for r in ds:
        try:
            sc = float(r["score"])
        except (TypeError, ValueError):
            continue
        lab = "positive" if sc > POS_T else "negative" if sc < NEG_T else "neutral"
        rows.append({"text": r["sentence"], "label": lab})
    return rows


def _llm_pred(text):
    prompt = ("Classify the financial sentiment of this sentence as exactly one word: positive, "
              f"negative, or neutral.\n\nSENTENCE: {text}\n\nAnswer:")
    try:
        r = llm.complete([{"role": "user", "content": prompt}], model=config.FAST_MODEL,
                         temperature=0.0).lower()
        m = re.search(r"\b(positive|negative|neutral)\b", r)
        return m.group(1) if m else "neutral"
    except Exception:
        return "neutral"


def _metrics(gold, pred):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    f1s = []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0)
    return acc, sum(f1s) / len(f1s)


def run(n=150):
    rows = _load_fiqa()
    random.Random(SEED).shuffle(rows)
    rows = rows[:n]
    texts = [r["text"] for r in rows]
    gold = [r["label"] for r in rows]
    print(f"FiQA held-out: {len(rows)} sentences (seed {SEED}); gold dist {dict(Counter(gold))}")

    fb = [s["label"] for s in sentiment.score(texts)]
    fb_acc, fb_f1 = _metrics(gold, fb)
    lm = [_llm_pred(t) for t in texts]
    lm_acc, lm_f1 = _metrics(gold, lm)

    lines = [
        f"FinBERT vs 8B-LLM on HELD-OUT FiQA (out-of-domain), n={len(rows)}",
        f"gold distribution: {dict(Counter(gold))}",
        "",
        f"FinBERT: accuracy {fb_acc*100:.0f}%  macro-F1 {fb_f1*100:.0f}%",
        f"8B LLM:  accuracy {lm_acc*100:.0f}%  macro-F1 {lm_f1*100:.0f}%",
        "",
        "Compare to the IN-DOMAIN PhraseBank result (FinBERT 98% / LLM 92%): if the gap here is",
        "similar, FinBERT's edge is real; if it shrank, the 98% was largely PhraseBank-specific and",
        "FinBERT's value is mainly determinism + being a trained DS model, not raw accuracy.",
        "Note: FiQA is mostly polar (few true neutrals) and bucketed from a continuous score, so",
        "absolute numbers are lower than PhraseBank for everyone - the FinBERT-vs-LLM GAP is the point.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
