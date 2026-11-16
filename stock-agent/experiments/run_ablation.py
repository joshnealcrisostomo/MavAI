"""Prompt Engineering ablation: run a fixed query set through each system-prompt variant and
compare tool-call correctness, ticker-resolution accuracy, latency, and token cost.

This is the experiment behind the Prompt Engineering module. Run from the project root:
    ./.venv/Scripts/python.exe experiments/run_ablation.py

It loads .env itself. Results print as a table and are written to experiments/results/.

CASES is split into two groups:
  - EASY_CASES: the original 6 queries (clear US ticker, clear intent, explicit-ish dates).
    All three variants hit 100% on these - they don't discriminate prompt quality.
  - HARD_CASES: added to actually separate the variants. These lean on things only the
    `rules`/`few_shot` prompts explicitly cover - PH stocks (the `minimal` prompt literally
    says "US stocks" only), relative/fuzzy dates, less-common tickers, and cross-market or
    two-PH-ticker comparisons.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

from core import agent, prompts  # noqa: E402

# query -> the tool we expect the agent to call and the tickers it should resolve
EASY_CASES = [
    {"q": "What were the min and max prices of Apple in Q1 2025?",
     "tool": "get_stock_history", "tickers": ["AAPL"]},
    {"q": "current price of Microsoft", "tool": "get_current_price", "tickers": ["MSFT"]},
    {"q": "Show Tesla's 20-day moving average over the last 3 months",
     "tool": "get_moving_average", "tickers": ["TSLA"]},
    {"q": "Compare Apple and Google this year and say which did better",
     "tool": "compare_stocks", "tickers": ["AAPL", "GOOG"]},
    {"q": "What were Nvidia's highest and lowest closing prices last month?",
     "tool": "get_stock_history", "tickers": ["NVDA"]},
    {"q": "How does Amazon's price compare to Walmart's year to date?",
     "tool": "compare_stocks", "tickers": ["AMZN", "WMT"]},
]

# harder queries, designed to actually separate minimal / rules / few_shot instead of all
# three scoring 100%
HARD_CASES = [
    # PH ticker on a prompt that tells the model it only covers "US stocks" (minimal) -
    # the single most likely case to make minimal fail while rules/few_shot succeed
    {"q": "What's the current price of Jollibee?",
     "tool": "get_current_price", "tickers": ["JFC"]},
    # PH ticker + a relative date range needing conversion
    {"q": "Globe's min and max over the last 3 months",
     "tool": "get_stock_history", "tickers": ["GLO"]},
    # fuzzy/relative date with no explicit range or quarter name
    {"q": "How has Nvidia performed over the past two weeks?",
     "tool": "get_stock_history", "tickers": ["NVDA"]},
    # less-common PH ticker, not in the few-shot worked examples
    {"q": "What's GT Capital's 30-day moving average over the last month?",
     "tool": "get_moving_average", "tickers": ["GTCAP"]},
    # two PH tickers compared against each other (never tested in the original set)
    {"q": "Compare Ayala Land and SM Prime year to date",
     "tool": "compare_stocks", "tickers": ["ALI", "SMPH"]},
    # cross-market comparison: one US, one PH ticker in the same call
    {"q": "Compare Apple and Jollibee this year",
     "tool": "compare_stocks", "tickers": ["AAPL", "JFC"]},
]

CASES = EASY_CASES + HARD_CASES


def first_tool_and_tickers(result):
    """Pull the first tool the agent actually called and the tickers it passed."""
    for r in result["tool_results"]:
        t = r.get("tool")
        if t in ("get_stock_history", "get_current_price", "get_moving_average"):
            return t, [r.get("ticker")]
        if t == "compare_stocks":
            return t, r.get("tickers", [])
    return None, []


def run_variant(name, prompt):
    rows, tool_hits, ticker_hits, tokens, latency = [], 0, 0, 0, 0.0
    for c in CASES:
        t0 = time.perf_counter()
        res = agent.run_agent(c["q"], system_prompt=prompt)
        dt = (time.perf_counter() - t0) * 1000
        tool, tickers = first_tool_and_tickers(res)
        tool_ok = tool == c["tool"]
        ticker_ok = set(tickers) == set(c["tickers"])
        tool_hits += tool_ok
        ticker_hits += ticker_ok
        tokens += sum((getattr(u, "total_tokens", 0) or 0) for u in res["usages"])
        latency += dt
        rows.append({"q": c["q"], "expected_tool": c["tool"], "got_tool": tool,
                     "tool_ok": tool_ok, "ticker_ok": ticker_ok})

    time.sleep(10)

    n = len(CASES)
    return {"variant": name, "tool_acc": tool_hits / n, "ticker_acc": ticker_hits / n,
            "avg_latency_ms": round(latency / n, 1), "avg_tokens": round(tokens / n, 1),
            "rows": rows}


def main():
    results = [run_variant(name, prompt) for name, prompt in prompts.VARIANTS.items()]

    print(f"\n{'variant':10} {'tool_acc':>9} {'ticker_acc':>11} {'avg_ms':>9} {'avg_tokens':>11}")
    print("-" * 54)
    for r in results:
        print(f"{r['variant']:10} {r['tool_acc']:>8.0%} {r['ticker_acc']:>10.0%} "
              f"{r['avg_latency_ms']:>9} {r['avg_tokens']:>11}")

    outdir = ROOT / "experiments" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outdir / 'ablation.json'}")


if __name__ == "__main__":
    main()