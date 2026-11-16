"""Deterministic financial-tone layer (FinBERT).

The agent otherwise has to eyeball a pile of headlines and guess the mood - non-deterministic and
easy to overload. FinBERT (ProsusAI/finbert) gives a fixed positive/negative/neutral score per
headline, so we can summarise a day's news into one reproducible signal the agent reasons over.

This is a DESCRIPTION of news tone, not a price prediction and not advice - net-negative news does
not reliably mean the price drops (see the forecast bake-off; direction is ~coin flip).

Lazy singleton like core.forecast's Chronos loader, so importing this module never loads torch.
"""
_pipe = None


def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, top_k=None)
    return _pipe


def score(texts: list[str]) -> list[dict]:
    """Per-text tone: [{label, positive, negative, neutral}] with class probabilities."""
    if not texts:
        return []
    out = []
    for r in _load()(texts):                      # top_k=None -> list of {label,score} per text
        d = {x["label"].lower(): float(x["score"]) for x in r}
        out.append({
            "label": max(d, key=d.get),
            "positive": round(d.get("positive", 0.0), 3),
            "negative": round(d.get("negative", 0.0), 3),
            "neutral": round(d.get("neutral", 0.0), 3),
        })
    return out


def aggregate(headlines: list[str]) -> dict:
    """Net tone over a list of headlines: counts + a net score in [-1,1] + a label."""
    if not headlines:
        return {"n": 0, "label": "no news", "net": 0.0, "positive": 0, "negative": 0, "neutral": 0}
    s = score(headlines)
    pos = sum(1 for x in s if x["label"] == "positive")
    neg = sum(1 for x in s if x["label"] == "negative")
    neu = sum(1 for x in s if x["label"] == "neutral")
    net = round((pos - neg) / len(s), 3)
    label = "positive" if net > 0.15 else "negative" if net < -0.15 else "mixed/neutral"
    return {"n": len(s), "positive": pos, "negative": neg, "neutral": neu, "net": net, "label": label}


def warmup():
    _load()
    return score(["Record profit sent shares soaring."])
