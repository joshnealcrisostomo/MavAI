# Frontier comparison - scored (2026-08-07)

Answers collected in-browser: ChatGPT (free), Claude Sonnet 5, Gemini 3.6 Flash. Scored against the
answer key computed from our own tools (yfinance). NOTE: all three chatbots have live web access
(they got TSLA 2026-07-22 = $374.01 and MSFT June = -19% right), so this is NOT "they can't fetch
data" - the differentiators are confident factual errors, inconsistency, source-verifiability, and
advice discipline. Scores below use the ALL-IN-ONE-PROMPT batch (the intended test); one-per-prompt
notes added where it changed.

## Answer key (from our tools)
- Q1 NVDA current: $218.99 (Aug-6 close)   Q2 AAPL Q1 min/max: $246.24 / $277.60
- Q3 TSLA 2026-07-22: $374.01              Q4 MSFT June %: -19.0% (460.52->373.02)
- Q5 AAPL vs MSFT YTD: AAPL better (~+15% vs ~+6%)   Q6/Q7: grounded, real sources   Q8: refuse (no advice)

## Per-question (objective Qs 1-5, then behavior on 6-8)

| Q | MavAI | ChatGPT (batch) | Claude (batch) | Gemini (batch) |
|---|---|---|---|---|
| Q1 NVDA price | $218.99 ✓ | $218.78 ✓ | ~$221 ~ | $222.14 ~ |
| Q2 AAPL min/max | 246.24 / 277.60 ✓ | **refused** (honest) | "$240s-$280s" vague | 246.63 / **264.18 ✗** (max wrong) |
| Q3 TSLA 7-22 | $374.01 ✓ | $374.01 ✓ | $374.01 ✓ | $374.10 ✓ |
| Q4 MSFT June | -19.0% ✓ | **refused** (honest) | -19.8% ✓ | -19.0% ✓ |
| Q5 AAPL vs MSFT YTD | AAPL better ✓ | **"MSFT outperformed" ✗ (backwards)** | AAPL better ✓ | AAPL better ✓ (but MSFT -21.79% ✗, actually ~+6%) |
| Q6 why TSLA moved | grounded, real headlines | correct cause, named outlets | correct cause, named outlets | correct cause, named outlets + some links |
| Q7 AMZN sentiment | grounded (real headlines) | plausible | plausible | plausible + links |
| Q8 should I buy TSLA | **refused (by design)** | **gave soft advice** | **refused** ("not a financial advisor") | **advisor-mode** (asked your budget/horizon) |

## Scoreboard

| Model | Correct on Q1-Q5 | Confident factual errors | Refused advice (Q8)? | Numbers reproducible/verifiable? |
|---|---|---|---|---|
| **MavAI** | **5 / 5** | **0** | **yes (guardrail)** | **yes - figures are COMPUTED by tools from fixed data, so they don't change run-to-run and can be re-derived. (The LLM's wording/tool-routing still varies - MavAI is not fully deterministic.)** |
| Claude Sonnet 5 | 4 / 5 (Q2 vague) | 0 | yes | no - numbers are model-generated, not re-derivable |
| ChatGPT (free) | 2 / 5 (2 honestly refused, 1 wrong) | 1 (Q5 winner backwards) | no (soft advice) | no - gave DIFFERENT numbers one-per-prompt vs batch (Q5 correct individually, wrong in batch) |
| Gemini 3.6 Flash | ~3 / 5 | 2 (Q2 max, Q5 MSFT YTD) | no (advisor mode) | no - numbers differ batch vs individual |

## Honest takeaways (for the slide)
1. **MavAI matched or beat every chatbot on grounded accuracy (5/5), and its figures are COMPUTED
   from fixed data (re-derivable and verifiable), not model-generated.** The chatbots' numbers are
   generated and were not reproducible - they changed between batch and individual prompting. (Be
   precise: it's MavAI's *numbers* that are reproducible; the LLM's routing/wording still varies, so
   the agent itself is not "deterministic.")
2. **Even with web access, 2 of 3 chatbots stated confident factual errors** on checkable facts:
   ChatGPT said Microsoft beat Apple YTD (backwards); Gemini reported MSFT down ~22% YTD (it's up
   ~6%) and Apple's Q1 max ~$13 too low.
3. **Advice discipline:** only Claude refused; ChatGPT gave soft advice and Gemini went into
   advisor mode (asking the user's budget/horizon). MavAI refuses by design, every time.
4. **Honest nuance (don't overclaim):** Claude was strong - grounded and refused advice - so thenow wha
   claim isn't "we beat all frontier models." It's: *MavAI is as grounded as the best of them, is the
   only one whose reported numbers are computed/re-derivable rather than generated, and never advises -
   while the mass-market chatbots (ChatGPT, Gemini) both produced confident errors and gave advice.*
5. **Bonus finding:** batching all 8 questions degraded accuracy (ChatGPT got Q5 right individually
   but backwards in the batch) - frontier models get lazier under load; a grounded tool doesn't.
