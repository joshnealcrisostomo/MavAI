"""Evaluate the FinBERT sentiment model on Financial PhraseBank, with an 8B-LLM baseline.

FinBERT (ProsusAI/finbert) is a deterministic financial-sentiment classifier: sentence -> positive
/ negative / neutral, same output every run. This is the deterministic "what's the tone" function
we want to summarise a day's news for the agent.

Financial PhraseBank is the standard labeled benchmark (financial sentences + gold sentiment).

IMPORTANT HONESTY CAVEAT (printed in the report too): FinBERT was fine-tuned ON Financial
PhraseBank, so this is largely an IN-DOMAIN check, not a generalisation number - expect it to score
high because it has effectively seen this data. That's why we also run the general 8B LLM on the
same sentences: the LLM was NOT trained on PhraseBank, so FinBERT-vs-LLM shows whether the
specialised model actually helps. For a true generalisation number you'd use a held-out set (e.g.
FiQA).

Writes experiments/results/finbert_sentiment.txt. Run from project root (optional sample size):
    ./.venv/Scripts/python.exe experiments/eval_finbert.py 150
"""
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm  # noqa: E402

OUT = ROOT / "experiments" / "results" / "finbert_sentiment.txt"
SEED = 42
LABELS = ["negative", "neutral", "positive"]


def _load_phrasebank():
    # gtfintechlab hosts PhraseBank as plain parquet (the new datasets lib rejects the old
    # script-based takala repo). Config '5768' is one seed's train/test of the all-agree subset.
    from datasets import concatenate_datasets, load_dataset
    repo, cfg = "gtfintechlab/financial_phrasebank_sentences_allagree", "5768"
    parts = []
    for sp in ("train", "test"):
        try:
            parts.append(load_dataset(repo, cfg, split=sp))
        except Exception:
            pass
    if not parts:
        raise RuntimeError("could not load Financial PhraseBank")
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return ds, LABELS, f"{repo}:{cfg} (all-agree)"  # LABELS index = 0 neg, 1 neutral, 2 positive


def _gold_name(v, names):
    if isinstance(v, str) and not v.isdigit():
        return v.lower()
    return names[int(v)].lower()


def _finbert_preds(sentences):
    from transformers import pipeline
    clf = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, top_k=1)
    out = []
    for i in range(0, len(sentences), 32):
        for r in clf(sentences[i:i + 32]):
            lab = (r[0]["label"] if isinstance(r, list) else r["label"]).lower()
            out.append(lab)
    return out


def _llm_pred(sentence):
    prompt = ("Classify the financial sentiment of this sentence as exactly one word: positive, "
              f"negative, or neutral.\n\nSENTENCE: {sentence}\n\nAnswer:")
    try:
        r = llm.complete([{"role": "user", "content": prompt}], temperature=0.0).lower()
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
    ds, names, src = _load_phrasebank()
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    idx = idx[:n]
    sents = [ds[i]["sentence"] for i in idx]
    gold = [_gold_name(ds[i]["label"], names) for i in idx]
    print(f"loaded {src}: {len(ds)} sentences; testing {len(sents)} (seed {SEED})")
    print("gold distribution:", dict(Counter(gold)))

    fb = _finbert_preds(sents)
    fb_acc, fb_f1 = _metrics(gold, fb)
    lm = [_llm_pred(s) for s in sents]
    lm_acc, lm_f1 = _metrics(gold, lm)

    lines = [
        f"FinBERT sentiment eval on Financial PhraseBank ({src}), n={len(sents)}",
        "",
        f"FinBERT (deterministic DS model): accuracy {fb_acc*100:.0f}%  macro-F1 {fb_f1*100:.0f}%",
        f"8B LLM baseline (llama-3.1-8b):    accuracy {lm_acc*100:.0f}%  macro-F1 {lm_f1*100:.0f}%",
        "",
        "CAVEAT: FinBERT was fine-tuned ON Financial PhraseBank, so its number here is IN-DOMAIN "
        "(effectively train-on-test) and overstates generalisation. The value of this eval is the "
        "COMPARISON: the 8B LLM was not trained on PhraseBank, so FinBERT beating it shows the "
        "specialised model genuinely helps on financial-sentiment tone. For a true generalisation "
        "number, evaluate on a held-out set like FiQA.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
