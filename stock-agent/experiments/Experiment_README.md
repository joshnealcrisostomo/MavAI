# Experiments

The core evidence for the write-up's Experiment Findings section comes from two scripts,
`eval_suite.py` and `run_ablation.py`. Everything else in this folder is supporting or optional -
dataset building, extra benchmarks, and one-off scans used to produce other pieces of evidence
(forecast backend choice, RRL findings, the curated benchmark, the frontier comparison).

`run_all.py` chains the core pipeline (and, on request, the optional extras) in one command.
Run everything from the project root (scripts load `.env` themselves):

```bash
./.venv/Scripts/python.exe experiments/run_all.py
```

## Core pipeline (run by default via `run_all.py`)

0. **check_keys.py** - pre-flight check that required API keys are set. Skip with
   `--skip-key-check`.
1. **build_dataset.py** - builds `results/forecast_dataset.json`, the shared input for
   `eval_forecast_vetting.py` and `eval_ragas.py`. Skipped if the file already exists; force a
   rebuild with `--rebuild-dataset`.
2. **eval_suite.py** - reliability suite. Four checks, reported as pass/fail with the actual
   values:
   1. **Tool accuracy** - the `get_stock_history` min/max/avg match an independent yfinance
      computation (proves displayed numbers are correct, not hallucinated).
   2. **Guardrails** - advice, off-topic, and prompt-injection inputs are blocked; real queries
      pass.
   3. **Disambiguation** - a missing ticker triggers a clarifying question; a named or
      context-implied ticker proceeds; concept questions don't ask for a ticker.
   4. **Edge cases** - reversed date ranges, bad tickers, and under-specified comparisons fail
      cleanly without crashing.
3. **eval_forecast_vetting.py** - unit / trajectory / end-to-end / LLM-judge tiers on the
   forecast-vetting agent.
4. **eval_ragas.py** - RAGAS Faithfulness + Answer Relevance on the same vetting answers, the
   standard-framework view of step 3's generation checks. Cap the sample with `--ragas-limit N`;
   regenerate cached vetting answers with `--ragas-rebuild`.

Any core step can be skipped: `--skip-suite`, `--skip-vetting`, `--skip-ragas`.

## Prompt engineering (run separately, not part of `run_all.py`)

```bash
./.venv/Scripts/python.exe experiments/run_ablation.py
```

Runs a fixed query set through three system-prompt variants (`minimal`, `rules`, `few_shot`) and
reports, per variant: tool-selection accuracy, ticker-resolution accuracy, average latency, and
average tokens. `CASES` is split into `EASY_CASES` (the original 6 queries - all three variants
hit 100%, so they don't discriminate prompt quality) and `HARD_CASES` (PH tickers, fuzzy dates,
less-common tickers, cross-market comparisons - added to actually separate the variants). Writes
`results/ablation.json`. This is the experiment that justifies Prompt Engineering as a module - it
shows which prompt design the live app should use and why. Slow (sleeps 10s between variants for
rate limits); available via `run_all.py --with-ablation` (also included in `--with-extras`) but
not run by default.

## Optional extras (opt in via `run_all.py` flags, or run standalone)

- **`--with-curated` / eval_curated.py** - scores MavAI against `data/curated/curated_events.csv`,
  a hand-labeled set of real news-driven price moves. Needs `curated_events.csv` filled in first,
  so it isn't meaningful to run blind on every invocation.
- **`--with-capability` / capability/*.py** - benchmarks the underlying skill (news sentiment,
  event/8-K extraction) against external datasets (EDT, FAANG 8-K, FinBERT PhraseBank + FiQA
  held-out), rather than this pipeline's own tiers.
- **`--with-bakeoff` / forecast_bakeoff.py** - walk-forward backtest of every forecast backend
  (naive/ets/arima/theta/chronos) on FAANG/Big Tech over 2026, no lookahead. This is the RRL
  evidence for Component #14: it shows every model sits near the naive baseline's ~50% coin flip
  on direction, which motivates the agent's news-vetting layer (the value is flagging *when* to
  distrust a forecast, not beating the market on price alone). Picks the recommended
  `FORECAST_BACKEND` and writes `results/forecast_bakeoff.txt`. Slow - fits every backend at every
  sampled as-of date across the ticker grid.
- **`--with-blindsided` / blindsided.py** - finds the "consistent-then-blindsided" showcase: an
  event where the price-only forecaster called several days right in a row, then got the event day
  wrong, and checks whether the vetting agent flagged it. One clean example of the value
  proposition in a single slide. Honesty caveat baked into the output: a short correct streak is
  well within chance over ~7 months of data, so this is illustrative, not proof the forecaster is
  predictive. Calls the vetting agent per candidate event, so cost scales with how many big moves
  `events.auto_events()` finds. Writes `results/blindsided.txt`.
- **`--with-scan-moves` / scan_moves.py** - reads 2026 price data (no AI calls) and lists the
  biggest up/down days per FAANG ticker, with gap% (overnight vs intraday) and same-day SPY% (to
  flag market-wide days vs company-specific ones), as a shortlist to hand-research for
  `curated_events.csv`. Writes `results/move_candidates.txt` and a starter
  `data/curated/move_candidates.csv`. Only useful when re-curating events, so it's cheap but
  opt-in. Takes optional args: `top_n` (default 8), `big_thresh` (default 4.0).
- **`--with-frontier` / frontier_benchmark.py** - puts a fixed question set (live prices,
  historical stats, why-did-it-move, advice-refusal) to MavAI and computes ground truth from our
  own tools, then writes `results/frontier_template.md` with blanks to manually paste in
  ChatGPT/Claude/Gemini's answers for a groundedness comparison. Only the ground-truth + MavAI
  half is automated - the chatbot-pasting step is manual - so it's opt-in rather than default.
- **`--with-why-feature` / eval_why_feature.py** - not yet documented here; see the script itself.
- **`--with-extras`** - shorthand for `--with-why-feature --with-frontier --with-bakeoff
  --with-ablation --with-scan-moves --with-blindsided` at once.

Individual reports land in `experiments/results/` (`eval_results.txt`, `forecast_vetting.txt`,
`ragas_eval.txt`, `curated_eval.txt`, `forecast_bakeoff.txt`, `blindsided.txt`,
`move_candidates.txt`, `ablation.json`, `frontier_template.md`), plus a combined
`all_evals_summary.txt` from `run_all.py`.