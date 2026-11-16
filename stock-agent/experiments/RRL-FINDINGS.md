# RRL / Model-Selection Findings (for the write-up + defense)

This is the Review-of-Related-Literature record: for each modeling choice we surveyed the options,
tested them, and recorded the finding and the decision. Raw numbers are in `experiments/results/`.
The theme throughout: we chose based on evidence, and we kept the honest negative results because
they are the point (they explain why the agent is built the way it is).

---

## 1. Forecasting model  (the required DS model, Component #14)

**Options surveyed:** naive random-walk-with-drift (baseline); classical statistical - AutoARIMA,
AutoETS, AutoTheta (via `statsforecast`); a time-series **foundation model**, Amazon **Chronos-Bolt**
(SOTA, zero-shot); deep learning (LSTM / N-BEATS) considered and rejected up front (data-hungry, would
overfit ~7 months of data, and pull heavy deps).

**Test:** walk-forward backtest on FAANG 2026, directional accuracy (HIT), `forecast_bakeoff.py`.

**Finding:** naive 50.9%, Theta 47.8%, Chronos 42.9%, ARIMA/ETS mostly abstain (predict flat 73-81% of
the time). **Nothing beat
the ~50% coin flip - not even the SOTA foundation model.** Consistent with the efficient-market
hypothesis.

**Decision:** ship **Theta** (a real statistical DS model, torch-free); keep Chronos as an
experiment-only comparison (it pulls torch and did not win). The "no model beats coin flip" result is
the justification for the whole vetting approach: the value isn't the forecast, it's judging it.

## 2. Sentiment model  (bonus second DS model - TESTED, NOT SHIPPED)

**Options:** specialized **FinBERT** (ProsusAI, fine-tuned on Financial PhraseBank) vs the general
**8B LLM** we already run.

**Test:** in-domain PhraseBank (`eval_finbert.py`), held-out FiQA (`eval_finbert_heldout.py`), and a
qualitative check on our own real Finnhub headlines.

**Finding:**
- In-domain (PhraseBank): FinBERT **98%** vs LLM 92% - looks great.
- Held-out (FiQA, out-of-domain): FinBERT **45%** vs LLM **71%** - the edge collapses; FinBERT even
  falls below a majority-class baseline.
- On our actual headlines: FinBERT and the LLM disagree ~46%, and in the clear cases (e.g. "Why MSFT
  Stock **Rose** Today", "as Stock **Sinks**") FinBERT is neutral-leaning and **wrong**; the LLM is right.

**Conclusion:** FinBERT's 98% was largely **in-domain memorization of PhraseBank's style**. On the
punchy headline text MavAI actually processes, it does **not** generalize and the LLM reads sentiment
better. Its only surviving advantage is determinism - not worth a torch dependency for a worse answer.

**Decision:** **We did not ship FinBERT.** We tried a specialized domain model, validated it out of
domain and on our own data, found it weaker than the general LLM, and left it out - keeping the eval
scripts as evidence. (Honest RRL story: testing beat assumption.)

## 3. Catalyst detection approach  (the beyond-midterm capability)

**Options:** LLM reading news to identify material events vs a keyword matcher.

**Test:** EDT ("Trade the Event") gold event labels, `eval_edt_events.py`, n=150.

**Finding:** LLM agent **93% accuracy, +87 discrimination**; keyword baseline only 73%. The LLM catches
events phrased without an obvious keyword (e.g. "FDA approval") and ignores calm docs that merely
contain a trigger word - it understands *materiality*, not just words.

**Decision:** use the LLM agent for catalyst detection. This is the dataset-backed, beyond-midterm
capability the Final rests on.

## 4. Is directional prediction even feasible?  (why we detect-and-flag, not predict)

**Test:** the bake-off (Section 1) + EDT news->direction (`eval_edt.py`).

**Finding:** every forecaster ~coin flip on direction; EDT news->direction 54% with 57% abstention.
Predicting the move is near-impossible for anyone.

**Decision:** the agent **detects catalysts and flags reliability, and refuses to predict or advise.**
Framed as the value proposition, not hidden as a limitation.

## 5. Historical news source

**Options:** yfinance / Google News RSS (live-only, no history); **GDELT** (free archive, but 1 req/5s
and full-text matching = noisy); **Finnhub** company-news (ticker-native, ~1yr history, needs a free key).

**Finding:** Finnhub is best but tags loosely (~40% of "symbol=NVDA" headlines actually mention Nvidia),
so a company-relevance filter is required. GDELT works as a keyless fallback but is slower/noisier.

**Decision:** Finnhub primary + relevance filter, GDELT fallback. This is now the SINGLE news-retrieval
layer (`core/news_hist.py`) for the whole app - both the vetting tool and the "why did it move"
explanation tool call it, and it routes PH tickers to Google News RSS. The Midterm's separate live-only
yfinance news feed was retired: it could not fetch a past window, so historical "why" queries always came
back empty. No-lookahead is applied by the caller, not the retriever: vetting cuts at the market open,
while the explanation tool reads same-day/intraday news (correct for explaining a past move, not
predicting it).

## 6. Ground truth for discrimination

**Options:** price-derived event/calm split (proxy - a big move isn't a real catalyst); EDT labeled
events (real, human-annotated); SEC 8-K filings (objective, covers 2026).

**Finding:** EDT gives a clean, dataset-backed discrimination number. The SEC-8-K check on FAANG 2026
came out **+0** - mega-caps are in the news *every day*, so per-day event/calm doesn't separate
(`faang_8k.txt`). A documented limitation, not a failure.

**Decision:** report discrimination on EDT (real labels); document the FAANG-per-day saturation.

## 7. Evaluation framework: RAGAS vs our own harness (Week-11 three-layer requirement)

**Context:** the Week-11 slides require all three eval layers (Unit / Trajectory / End-to-End) and name
**RAGAS** as a recognized framework. We already implement all three by hand in `eval_forecast_vetting.py`
(no-lookahead unit checks, grounding/trajectory, catch-rate end-to-end, LLM-judge). Question was whether
to add RAGAS.

**What we did:** added `experiments/eval_ragas.py` running RAGAS's two generation metrics that fit a
**tool-use** agent with **no hand-written reference answers** - **Faithfulness** (are the answer's claims
backed by what the agent was handed?) and **Answer Relevance** (does it answer the question?). Scored on
the 28-event FAANG-2026 vetting set; judge = 8B via Groq, embeddings = FastEmbed (torch-free).

**Honest scope note (important):** RAGAS is built for document-RAG. Its retrieval metrics (Context
Precision/Recall) assume gold-referenced passage chunks, which is a forced fit for an agent whose
"context" is tool output, so we **deliberately omit them**. Faithfulness fits only once "context" is
defined as *everything the agent was handed* - the forecast facts AND the headlines, not the headlines
alone (scoring against headlines alone wrongly flags true forecast-number claims as hallucinations; we
hit exactly that artifact and fixed it). RAGAS metrics are themselves LLM-judged, so they carry the same
judge caveats as our own LLM-judge tier: a scalable proxy, not ground truth.

**Result (28 events, 70B judge):** Answer Relevance **0.61** (0.68 excluding hedged 'caution' answers
RAGAS flags noncommittal), Faithfulness **0.43**. Raw numbers in `experiments/results/ragas_eval.txt`.

**Honest reading of the low faithfulness (this is itself a finding):** RAGAS Faithfulness checks that each
atomic claim is *entailed* by the context - it was built for extractive QA. Our vetting answers are
**analytical**, and most contain interpretive or **negative** claims ("headlines are routine", "no material
catalyst", "the forecast looks reasonable") that cannot be entailed from a list of headlines, so they score
unfaithful even when correct (e.g. TSLA 2026-02-20 scores 0.00 for a sound "no shock in the news" answer -
you cannot entail an *absence* from a headline list). So ~0.43 is a genre effect, not hallucination. The
meaningful grounding guarantee for this agent is the **code-enforced** one in `eval_forecast_vetting.py`
(TRAJECTORY tier): 100% of cited URLs are real pre-event headlines - the agent literally cannot cite a source
it was not handed.

**Decision:** keep our hand-rolled 3-layer suite as the primary evidence; report RAGAS Answer Relevance +
Faithfulness as the **standard-framework view of the Layer-1 generation checks**, WITH the interpretation
above. The lesson (good RRL material): off-the-shelf RAG metrics don't transfer cleanly to a tool-use,
analytical agent - Answer Relevance does, Faithfulness needs interpretation, and the retrieval metrics don't
apply at all.

## 8. Data granularity: intraday catalysts (limitation + future work)

**The limitation.** Our price data is DAILY (yfinance close bars). The live `vet_forecast` tool therefore
cuts the news window at the market OPEN (~09:30 ET) of the as-of day and never reads later items. That
cleanly captures overnight / pre-market catalysts (earnings after the close, pre-open releases - the
driver of most big single-day gaps), but it MISSES intraday catalysts: news that breaks during the
session and moves the stock after the open (an analyst note, a mid-day headline, a CEO comment).

**Why the open-cut is the honest line on daily data.** With one daily candle you cannot separate "the
1pm news" from "the 1:30pm move" - they are the same bar. If we fed the agent same-day intraday news and
then scored it against that same day's move, we would be letting it read news OF the move as it happens to
"predict" the move. That is lookahead / circularity. The market open is the only mechanical, reproducible
no-lookahead boundary that daily data supports: everything knowable before the trading day, judged against
what happens during/after it.

**Why the curated benchmark (`eval_curated.py`) is NOT limited by this.** In the curated benchmark the
no-lookahead filter is the HUMAN, not a code timestamp: we only include a headline we have personally
verified was published before the move. So an intraday catalyst (say a 1pm headline before a 1:30pm move)
CAN be included, because a human checked its timestamp - something the automated daily-data cut cannot do
at scale. Two more reasons it stays clean: (a) the judge never sees the realized outcome - `vetting.judge`
is shown only the forecast + the news, and the actual move is used only for reporting - so intraday news
does not leak the future into the reasoning; (b) the REASONING-ONLY mode tests recognition ("given this
verified pre-move catalyst, do you flag it?"), not prediction, which is a fair test even on intraday cases.
The FULL-PIPELINE mode still uses the open-cut, so the gap between the two modes MEASURES the intraday
retrieval limitation instead of being invalidated by it. (Integrity depends on curation discipline: only
genuinely pre-move headlines, timestamps verified, mind the PH/US timezone - a post-move "why it fell"
article would be lookahead.)

**Future work.** Switching to an intraday price source (Polygon.io, Tiingo, Finnhub candles, Databento)
would give a real timestamp for WHEN a move happened, so we could cut news right before it and score the
outcome from there - making intraday-driven moves testable automatically, and enabling a genuine product
feature: mid-session "is this forecast still reliable right now?" queries. Costs to weigh: it needs a
paid-ish API (free intraday history is thin), a rule for defining "the move" intraday (a daily -6% may be
one sharp drop or an all-day bleed), and the forecaster would stay daily (intraday forecasting is even
noisier). Deliberately deferred: marginal gain for the deliverable, real cost under time pressure.

---

### One-paragraph RRL summary (drop into the slide)
We surveyed forecasting models from naive baselines through classical statistics to a SOTA time-series
foundation model (Chronos), and found none beat a coin flip on direction - so we ship a simple,
transparent Theta model and build the value in an agentic vetting layer. We surveyed sentiment models
(specialized FinBERT vs a general LLM) and, after held-out and on-domain testing, found FinBERT's
benchmark edge was in-domain only and did not generalize to our headlines, so we did not ship it. Our
catalyst-detection capability (LLM over news) beats a keyword baseline 93% vs 73% on the EDT benchmark.
The honest through-line: directional prediction is infeasible, so MavAI's contribution is grounded,
tested reliability - detecting and flagging, never predicting or advising.

---

## References (Review of Related Literature)

VERIFY each citation (open it, confirm author/year/venue) before presenting - do not cite a source you
have not personally checked. Grouped by the design decision it supports.

Direction is near-unpredictable -> detect-and-flag, don't predict:
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of
  Finance, 25(2), 383-417.
- Malkiel, B. G. (1973). A Random Walk Down Wall Street. W. W. Norton.

Ship a simple statistical model (Theta), not complex ML/DL:
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning
  forecasting methods: Concerns and ways forward. PLOS ONE, 13(3): e0194889. (ML did not beat
  statistical methods on M3.)
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series
  and 61 forecasting methods. International Journal of Forecasting, 36(1), 54-74.
- Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a decomposition approach to
  forecasting. International Journal of Forecasting, 16(4), 521-530. (The model we ship.)

Rejecting data-hungry deep learning on ~7 months of data:
- Elsayed, S., Thyssens, D., Rashed, A., Jomaa, H. S., & Schmidt-Thieme, L. (2021). Do We Really Need
  Deep Learning Models for Time Series Forecasting? arXiv:2101.02118.

SOTA time-series foundation model we surveyed:
- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815 (Amazon).

Tooling / reference for ARIMA / ETS / Theta:
- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
  https://otexts.com/fpp3/
- Nixtla StatsForecast. https://github.com/Nixtla/statsforecast

Verify-before-use (contents not fully confirmed):
- ML-for-stock-prediction survey, Financial Innovation, DOI 10.1186/s40854-019-0157-x.
