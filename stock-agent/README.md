# MavAI Stock Price Agent

MavAI judges whether a short-term stock forecast should be trusted, by grounding that judgment in
real pre-event news the forecaster couldn't see - it does not try to predict prices well. The
forecaster (AutoTheta) is the required domain model, not the product: the bake-off proved no
backend beats a coin flip on direction, so the agent's actual job is producing a cited,
code-grounded trust/caution/reject verdict, not a better number. That vetting capability sits on
top of a retained general price/news assistant - natural-language questions about **US and
Philippine (PSE) stock prices** (min/max/average over a range, current price, moving averages,
multi-stock comparisons) - backed by live market data, with memory for follow-ups, guardrails, and
charts. A unified data layer picks the source automatically per ticker: **yfinance** for US
stocks, **phisix** for Philippine stocks - so the same tools answer for both markets.

> STAI100 Final Capstone. Built on a shared `core/` pipeline exposed through a Streamlit chat
> UI and a FastAPI REST endpoint, with MLflow + JSON-line LLMOps logging, packaged in a single
> Docker image.

Example questions it handles:

- "Should I trust the forecast for Tesla right now?" (runs AutoTheta, pulls pre-event news, returns a cited trust/caution/reject verdict - the core capability)
- "What were the min and max prices of Apple in Q1 2025?"
- "Compare Apple, Microsoft and Google year to date - which did best?"
- "Show me Tesla's 20-day moving average for the last 3 months."
- "What's the current price of NVIDIA?" → "How did it do last month?" (memory carries the ticker)
- "What's the current price of Globe?" / "How is Jollibee doing today?" (routes to PH data)
- "What is a moving average?" (answered from the reference store, no tool call)

## Architecture

```
                         ┌──────────────┐        ┌──────────────┐
                         │ Streamlit UI │        │  FastAPI API │   two channels,
                         │  + Plotly    │        │  REST + SSE  │   one pipeline
                         └──────┬───────┘        └──────┬───────┘
                                └──────────┬────────────┘
                                           ▼
                                  core/pipeline.py
                                           │
   ┌───────────────┬───────────────┬───────┴───────┬───────────────┬──────────────┐
   ▼               ▼               ▼               ▼               ▼              ▼
 Guardrails   Disambiguation     RAG          ReAct Agent       Memory       Observability
 (validate/   (missing-ticker  (ticker +    (70B tool-calling  (summary    (MLflow + JSON
  disclaimer)  clarify, Wk2     glossary      loop; records a    buffer,     line per request:
               slot-filling)    reference)    thought/action/    carries     latency/tokens/cost)
                                              observation trace)  ticker)
                                                   │
                                                   ▼
                                              core/tools.py
        ┌──────────────────────────────────────────┴───────────────────────────┐
        ▼                                                                        ▼
  MARKET TOOLS ── yfinance/phisix ──► live prices         vet_forecast  CV/DS
  get_stock_history · get_current_price ·                        │  (the core, non-decorative
  get_moving_average · compare_stocks ·                          │   domain model of the app)
  get_price_move_context (why-did-it-move)                       ▼
                                                 ┌───────────────────────────────────┐
                                                 │ core/forecast.py  ── DS MODEL      │
                                                 │ AutoTheta (statsforecast), the     │
                                                 │ bake-off winner; torch-free.       │
                                                 │ naive / ARIMA / ETS / Chronos-Bolt │
                                                 │ surveyed + compared in the RRL     │
                                                 └────────────────┬──────────────────┘
                                                                  ▼
                                                 ┌───────────────────────────────────┐
                                                 │ core/vetting.py  ── LLM AUDITOR    │
                                                 │ judges forecast against pre-event  │
                                                 │ news (Finnhub→GDELT, no-lookahead) │
                                                 │ → ForecastVerdict:                 │
                                                 │   trust / caution / reject         │
                                                 │ sensitivity: investor (precise) |  │
                                                 │   analyst (wide net) — measured    │
                                                 │   trade-off, see §4.7 write-up     │
                                                 └────────────────┬──────────────────┘
                                                                  ▼
                                       Structured Outputs (Pydantic) - typed response + chart spec;
                                       numbers come from tools/DS model, never model recall
```

Per turn: **input guardrails → disambiguation (maybe ask) → RAG reference → ReAct agent (calls
tools; for a reliability question it calls `vet_forecast`, which runs the DS forecaster then the LLM
auditor) → typed response + chart spec → output disclaimer → save memory → LLMOps log.**

**Where the CV/DS model plugs in (Component #14):** the DS forecaster (`core/forecast.py`, AutoTheta)
is wrapped as the `vet_forecast` tool the ReAct agent calls. Its forecast is then vetted by the LLM
auditor (`core/vetting.py`) against no-lookahead news. So the DS model is a first-class step in the
agent pipeline, not a decorative add-on. Evaluated by `experiments/` (bake-off / RRL, Week-11
3-layer suite, RAGAS, the hand-curated benchmark, EDT catalyst detection, and the frontier comparison).

**News retrieval is one shared layer** (`core/news_hist.py`, with `core/gdelt.py` as the keyless
fallback client): both the vetting tool and the "why did it move" explanation tool pull company
news from it (US -> Finnhub primary/GDELT DOC 2.0 fallback, PH -> Google News RSS). Results are
disk-cached for offline-reproducible backtests. No-lookahead is applied by the caller, not by the
retrieval layer itself - vetting cuts news at the market open and keeps only the top 15
overnight/pre-market items the forecaster could not have seen; the explanation tool reads
same-day/intraday news, since it is explaining a past move rather than predicting one.

## Modules (8 modules, ≥2 per team member)

| # | Module | Where | What it does |
|---|--------|-------|--------------|
| 1 | Tool Use | `core/tools.py`, `core/market_data.py`, `core/ph_data.py` | Four market-agnostic tools (history, current, MA, compare) over a unified layer routing to yfinance (US) or phisix (PH) |
| 2 | ReAct Agent | `core/agent.py` | 70B tool-calling reason→act loop; surfaces a thought/action/observation trace |
| 3 | Structured Outputs | `core/schemas.py` | Pydantic `StockQuery`/`StockResponse` + validate-and-retry helper (Week 2 idiom) |
| 4 | Guardrails | `core/guardrails.py` | Blocks off-topic/injection and most advice-seeking phrasings via keyword + LLM checks; appends a financial disclaimer to every answer. Note: implicitly-phrased advice requests can currently evade detection — see Known limitations. |
| 5 | Disambiguation | `core/disambiguation.py` | Detects a missing stock and asks; slot-filling (Week 2 idiom) |
| 6 | Memory | `core/memory.py` | `ConversationSummaryBufferMemory`; carries the ticker across follow-ups |
| 7 | RAG | `core/rag.py` | Chroma store of ticker→company facts + a finance glossary (ONNX MiniLM, no torch) |
| 8 | Prompt Engineering | `core/prompts.py` + `experiments/run_ablation.py` | Three prompt variants, ablated and compared |

Plus the required deliverables: **Chat UI** (`app_streamlit.py`), **API Endpoint** (`api.py`),
**LLMOps Monitoring** (`core/observability.py`), **Dockerization** (`Dockerfile`).

> Module ownership across the 3 team members:
- Patrick Lim - Dockerization, ReAct Agent, API, & Tool Use
- Josh Neal Crisostomo - Disambiguation, Guardrails, Memory, Prompt Engineering, & Structured Outputs
- John Maverick Cisneros - RAG, Prompt Engineering, LLMOps Monitoring, API, Dockerization, & Chat UI

## Setup

### Local (Python 3.12+)

```bash
cd stock-agent
python -m venv .venv
# Windows:  .venv\Scripts\activate     macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then put your free Groq key in .env (https://console.groq.com/keys)

# Streamlit UI  -> http://localhost:8501
streamlit run app_streamlit.py

# FastAPI       -> http://localhost:8000/docs   (run in a second terminal)
uvicorn api:app --reload
```

### Docker (single image, both channels on one port)

```bash
cp .env.example .env      # add your Groq key
docker compose up --build
# Streamlit UI    -> http://localhost:8080
# FastAPI Swagger -> http://localhost:8080/api/docs
# health          -> http://localhost:8080/api/health
```

## API

```bash
# one-shot JSON
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "min and max of Apple in Q1 2025?", "session_id": "demo"}'
```

Response: `answer`, resolved `tickers`, a `chart` spec, `tools_called`, the ReAct `trace`, and
`session_id` (reuse it on the next call so memory carries over). `POST /api/chat/stream` returns
the same payload over Server-Sent Events.

## Experiments

The eval suite grew well past the Midterm's reliability check. `experiments/run_all.py` chains
the default tiers end to end (pre-flight key check → dataset build → reliability suite →
forecast-vetting unit/trajectory/e2e/judge tiers → RAGAS), collects pass/fail per step, and writes
`experiments/results/all_evals_summary.txt`; it exits non-zero if anything failed, so it's CI-friendly.
Slower or opt-in analyses (frontier comparison, forecaster bake-off, prompt ablation, EDT/FinBERT
capability benchmarks, the hand-curated benchmark, event-scanning utilities) are behind flags so a
default run stays fast:

```bash
./.venv/Scripts/python.exe experiments/run_all.py                 # default tiers (see above)
./.venv/Scripts/python.exe experiments/run_all.py --with-curated  # + hand-labeled 77-row benchmark
./.venv/Scripts/python.exe experiments/run_all.py --with-extras   # truly everything (slow)

# individual scripts still run standalone, e.g.:
./.venv/Scripts/python.exe experiments/eval_suite.py       # reliability: tool/guardrail/routing checks
./.venv/Scripts/python.exe experiments/run_ablation.py     # prompt-variant comparison
./.venv/Scripts/python.exe experiments/forecast_bakeoff.py # walk-forward FAANG backtest (naive/ARIMA/ETS/AutoTheta/Chronos-Bolt)
./.venv/Scripts/python.exe experiments/frontier_benchmark.py  # MavAI vs ChatGPT/Claude/Gemini groundedness
```

What each tier is measuring, briefly:

- **Reliability suite** - tool accuracy, guardrails, disambiguation, edge cases, market routing (US/PH), cross-market comparisons.
- **Forecast-vetting tiers + RAGAS** - unit/trajectory/e2e/LLM-judge checks on `vet_forecast`, plus RAGAS Faithfulness/Answer Relevance over the same vetting answers.
- **Forecaster bake-off** - justified AutoTheta as the shipped DS backend; every backend tested, including a SOTA foundation model, is near-random on direction.
- **EDT catalyst detection** - 93% event-detection accuracy against hand-labeled gold events vs. a 73% keyword baseline; direction-from-news came out near a coin flip.
- **Hand-curated benchmark** (77 rows, no AI in the labeling) - quantifies the investor/analyst sensitivity trade-off (high-precision/low-recall vs. wide-net) and news-retrieval recall (~9%).
- **Frontier benchmark** - MavAI vs. ChatGPT/Claude/Gemini on 8 identical FAANG questions; MavAI was the only one with reproducible, tool-computed numbers.

See `experiments/README.md` for the full list of scripts.

## Configuration

All via env vars (see `.env.example`): `GROQ_API_KEY`, `AGENT_MODEL` (default
`llama-3.3-70b-versatile`, runs the agent), `FAST_MODEL` (default `llama-3.1-8b-instant`, runs
the cheap classifiers). The two-model split is deliberate - the big model for reliable
tool-calling, the small one for high-volume classification.

## Known limitations

- **Forecast-vetting scope is narrow by design.** `vet_forecast` currently covers US FAANG/Big-Tech
  tickers (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA) over Jan 2026-present, where daily
  yfinance data and ticker-native Finnhub/GDELT news are both reliable. The general price/news
  assistant keeps the full US + PH scope.
- **Directional forecasting is near-random, on purpose.** Every backend surveyed in the bake-off
  (naive, ARIMA, ETS, AutoTheta, Chronos-Bolt) hovers near a coin flip on direction - this is the
  documented rationale for shipping a vetting layer instead of a predictor, not a bug to fix.
- **News-retrieval recall is low (~9% in the hand-curated benchmark).** The pipeline often doesn't
  surface the exact pre-event catalyst a human curator would cite, which caps the achievable
  catch-rate in both sensitivity modes - see `experiments/eval_curated.py` and write-up §4.7.
- **Sensitivity is a calibrated trade-off, not a solved setting.** Investor mode (precise) rarely
  false-flags calm days (95% calm-day trust) but misses most real events; analyst mode (wide net)
  catches every event but flags calm days too, collapsing its discrimination to ~0. Pick per use case.
- **No candlesticks for PH stocks.** US data (yfinance) includes full OHLC, so US stocks can be
  drawn as candlesticks. Philippine data (phisix) is close + volume only, so PH stocks get line
  charts, min/max/average, and moving averages - but not candlesticks. Everything else works for
  both markets. See `DECISIONS.md`.
- **First PH range query is slower.** PH history is fetched one trading day at a time and cached,
  so a fresh range takes a few seconds; repeat queries are instant. Pre-warm your demo range.
- **Not financial advice.** The agent reports data and refuses buy/sell/prediction requests.
- **Advice-seeking evasion.** Implicitly-phrased recommendation requests (e.g. "if you had to pick
  one, would you go with Tesla or Ford?") can currently bypass both guardrail checks: the
  keyword filter doesn't match the phrasing, and the topic classifier isn't responsible for advice
  detection. Found during the harder eval-suite test cases; see the technical write-up §4.1.
- **Ambiguous pronoun resolution.** If two tickers were mentioned earlier in a conversation, a
  follow-up like "how did it do last month?" may resolve to one of them rather than triggering a
  clarifying question - the disambiguation classifier is currently biased toward resolving over
  flagging genuine ambiguity.
- **Cross-market comparison check (open issue).** A comparison between a US and PH stock (e.g.
  Apple vs. Jollibee) correctly invokes the right tool and produces a correct-looking answer, but
  fails an automated eval check for reasons not yet root-caused.
- **Groq free tier** occasionally rejects a tool call; the agent retries up to 3 times with a
  slight temperature bump, and otherwise degrades to a "please rephrase" message rather than
  crashing.