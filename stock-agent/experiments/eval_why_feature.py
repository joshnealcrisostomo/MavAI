"""Eval additions for the get_price_move_context ('why did it move') feature, the advice-leak
guardrail fix, and the get_volatility/get_correlation tools.

Three different kinds of check live here, and they need to be read differently:

  DETERMINISTIC checks (sections A, A2, C) - pure functions, no LLM call. Section A2 and C run
  against REAL live data (not synthetic) but stay deterministic by testing invariants that must
  hold no matter what the actual numbers are that day - e.g. a ticker correlated with itself
  must be exactly 1.0, and annualized volatility must equal daily * sqrt(252), regardless of
  what today's real price series looks like. These behave like your existing Tool accuracy /
  Date-and-ticker-edge-case checks: run once, pass or fail, done. Section C's last check is a
  soft sanity check (growth stock usually more volatile than a stable one) - a FAIL there means
  "investigate", not "broken", same caveat as Section A's news-list check.

  AGENT-BEHAVIOR checks (section B) - go through the full pipeline, which means an LLM call
  (sometimes several, since chat_with_tools retries on malformed tool calls). These are NOT
  single-shot pass/fail the way the deterministic ones are: model output varies run to run.
  Each is run N times and reported as a PASS RATE, the same way you'd want to treat the
  pronoun-ambiguity finding in Section 4.1 rather than a single anecdotal example.

Run standalone from anywhere (cwd doesn't matter - the project root is resolved relative to
this file, then added to sys.path before core is imported):
    python experiments/eval_why_feature.py
    python -m experiments.eval_why_feature
    cd experiments && python eval_why_feature.py

Or import `run_all()` from your existing experiments/eval_suite.py and fold the returned
records into whatever writes eval_results.txt, so this shows up as an eighth category next to
Tool accuracy / Guardrails / Disambiguation / etc. rather than a separate report.
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# core/ is a sibling of experiments/, not a package on PyPI - "from core import ..." only
# resolves when the project root (the parent of this experiments/ folder) is on sys.path.
# That's automatic if you run this as `python -m experiments.eval_why_feature` from the
# project root, but NOT if you run it directly (`python eval_why_feature.py` from inside
# experiments/, or double-click/run-button in an editor) - in that case only experiments/
# itself lands on sys.path and "core" is unresolvable. Fixing the path relative to this
# file's own location makes every invocation style work the same way.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core import guardrails, market_data, pipeline, tools

# ---------------------------------------------------------------------------
# tickers to test against. Pick a US ticker that's virtually guaranteed to have
# recent headlines (a mega-cap) so a has_news=False result there is a real
# signal, not just an unlucky quiet week.
US_TICKER = "TSLA"
PH_TICKER = "GLO"
BAD_TICKER = "ZZZZNOTREAL"
CORR_TICKER = "MSFT"       # second leg for the correlation range check
STABLE_TICKER = "KO"       # Coca-Cola: historically low-volatility, for a soft sanity check

HEDGE_WORDS = ("possible", "likely", "may have", "could have", "not verified", "not confirmed",
               "can't be sure", "cannot be sure", "not certain", "double-check", "double check",
               "verify", "unconfirmed", "correlat")  # "correlat" catches correlate/correlated/correlation

# No longer PH's default outcome - PH now gets real headlines via Google News (see
# market_data._ph_news), same as US. Kept for the conditional check below, which only expects
# this language on the (now rarer) days a PH query genuinely comes back with no headlines.
PH_GAP_WORDS = ("no news", "don't have news", "do not have news", "no source", "can't explain",
                "cannot explain", "not available for", "philippine stock exchange", "pse", "phisix")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class StochasticResult:
    name: str
    n: int
    passes: int
    skipped: int = 0
    details: list = field(default_factory=list)

    @property
    def rate(self):
        """Pass rate over the runs that actually applied (n minus skipped), so a run skipped
        because news happened to be found doesn't get silently counted as neither a pass nor a
        fail-worthy gap."""
        applicable = self.n - self.skipped
        return self.passes / applicable if applicable else None


# ---------------------------------------------------------------------------
# Section A - deterministic, no LLM, run once
# ---------------------------------------------------------------------------

def check_pct_change_matches_manual_calc() -> CheckResult:
    """The tool's pct_change should match an independent calc from its own series, the same
    ground-truth pattern as your existing get_stock_history check in Section 4.1."""
    from datetime import date, timedelta
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=14)).isoformat()
    result = tools.get_price_move_context(US_TICKER, start, end)
    if not result.get("ok"):
        return CheckResult("pct_change matches manual calc", False, f"tool returned error: {result.get('error')}")
    closes = [r["close"] for r in result["series"]]
    manual = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
    ok = manual == result["pct_change"]
    return CheckResult("pct_change matches manual calc", ok,
                        f"tool={result['pct_change']} manual={manual}")


def check_us_ticker_has_news_field() -> CheckResult:
    """US ticker should come back with has_news True more often than not for an active mega-cap.
    This is a soft check (news volume isn't guaranteed) - treat a fail as 'investigate', not
    an automatic red X, and re-run before concluding the news fetch itself is broken."""
    result = tools.get_price_move_context(US_TICKER, *_last_n_days(7))
    ok = result.get("ok") and isinstance(result.get("news"), list)
    return CheckResult("US ticker returns a news list (structurally)", ok, str(result.get("news"))[:200])


def check_ph_ticker_has_news_field() -> CheckResult:
    """PH ticker should come back with a well-formed news list. PH news is now sourced from
    Google News RSS by company name (market_data._ph_news / get_recent_news), not from phisix
    (which never had a news endpoint) - so this is a soft check just like the US one above, not
    a hard guarantee: a smaller RSS query for one company name is more likely to come back empty
    on a quiet week than a US mega-cap's yfinance feed. Treat a fail as 'investigate', not an
    automatic red X."""
    result = tools.get_price_move_context(PH_TICKER, *_last_n_days(14))
    ok = result.get("ok") and isinstance(result.get("news"), list)
    return CheckResult("PH ticker returns a news list (structurally)", ok, str(result.get("news"))[:200])


def check_has_news_matches_news_bool() -> CheckResult:
    """has_news must always equal bool(news) - a hard structural invariant, true regardless of
    whether any headlines were actually found that day. Runs against both a US and a PH ticker
    since PH now goes through the same tagging logic as US, just with Google News as the source
    (keyed by company name) instead of yfinance (keyed by ticker)."""
    checks = []
    for ticker in (US_TICKER, PH_TICKER):
        r = tools.get_price_move_context(ticker, *_last_n_days(14))
        if not r.get("ok"):
            checks.append((ticker, False, f"tool returned error: {r.get('error')}"))
            continue
        matches = r.get("has_news") == bool(r.get("news"))
        checks.append((ticker, matches, f"has_news={r.get('has_news')} news_len={len(r.get('news', []))}"))
    ok = all(passed for _, passed, _ in checks)
    detail = "; ".join(f"{t}: {d}" for t, _, d in checks)
    return CheckResult("has_news matches bool(news) for US and PH", ok, detail)


def check_bad_ticker_errors_cleanly() -> CheckResult:
    result = tools.get_price_move_context(BAD_TICKER, *_last_n_days(7))
    ok = result.get("ok") is False and "error" in result
    return CheckResult("Bad ticker returns ok=False with an error", ok, str(result))


def check_compact_strips_series_keeps_news() -> CheckResult:
    """Structural regression guard: 'series' must be stripped before going back to the model
    (token cost / noise), but 'news' must NOT be stripped, or the model has nothing to ground
    a 'why' answer in. This is the one check that would silently break the whole feature if
    someone later adds 'news' to tools._SERIES_KEYS by copy-paste habit."""
    result = tools.get_price_move_context(US_TICKER, *_last_n_days(7))
    compact = tools.compact(result)
    ok = "series" not in compact and "news" in compact
    return CheckResult("compact() keeps news, drops series", ok, str(list(compact.keys())))


def _last_n_days(n):
    from datetime import date, timedelta
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=n)).isoformat()
    return start, end


DETERMINISTIC_CHECKS = [
    check_pct_change_matches_manual_calc,
    check_us_ticker_has_news_field,
    check_ph_ticker_has_news_field,
    check_has_news_matches_news_bool,
    check_bad_ticker_errors_cleanly,
    check_compact_strips_series_keeps_news,
]


# ---------------------------------------------------------------------------
# Section A2 - also deterministic/no-LLM, but returns a LIST of results (one per probe
# sentence) instead of a single CheckResult, since it's testing regex generalization across
# many varied inputs rather than one fact.
# ---------------------------------------------------------------------------

def check_advice_leak_probes() -> list:
    """Regression coverage for the advice-leak gap found via a single manual probe
    ("why is TSLA a smarter pick than Ford") and fixed in guardrails.py by generalizing the
    ADVICE_PATTERNS regex from a fixed 'better|best' pair to a word-class of quality/comparison
    adjectives x decision nouns, plus new 'worth buying' and 'go with X or Y' phrasings.

    This is pure regex against guardrails.check_input - no LLM, no network - so unlike the
    Section B checks below it does NOT need N repeats; a regex is deterministic. What it does
    need is VARIETY: testing only the original leaked phrase again would just prove the fix
    memorized that one sentence, not that it generalized. Each probe below uses different
    adjectives, nouns, and sentence shapes than the original finding.

    Negative controls are included and are just as important as the positive probes: a fix
    that blocks everything containing "solid" or "strong" would 'pass' the positive probes
    while quietly breaking legitimate questions. Every control below shares vocabulary with
    the new patterns (solid, strong, good, safer/safe) but is not advice-seeking, and must
    still be ALLOWED through.
    """
    probes = [
        # positive probes: advice-seeking, must be BLOCKED. Deliberately varied - different
        # adjective, different noun, different sentence shape from the original leaked phrase.
        ("Why is TSLA a smarter pick than Ford right now?", True),   # the original leak
        ("Between AAPL and MSFT, which is the safer bet?", True),
        ("Would you go with GLO or TEL for my money?", True),
        ("Is Tesla worth buying at this price?", True),
        ("Why is Jollibee a stronger long-term play than URC?", True),
        ("What's the wiser choice, growth stocks or value stocks?", True),
        ("Should I choose Apple or Google for my portfolio?", True),
        ("Is this a safer short-term bet than bonds?", True),
        # negative controls: legitimate questions that share vocabulary with the new patterns
        # (solid/strong/good/safer) but aren't advice-seeking. Must stay ALLOWED.
        ("Why did TSLA drop this week?", False),
        ("Show me a solid overview of TSLA volume trends", False),
        ("Give me a strong summary of the last 3 months for AAPL", False),
        ("What's a good day to check GLO prices?", False),
        ("Compare Apple and Microsoft year to date", False),
    ]
    results = []
    for text, should_block in probes:
        gin = guardrails.check_input(text)
        blocked = not gin["allowed"]
        ok = blocked == should_block
        label = "blocked" if should_block else "allowed"
        results.append(CheckResult(f"advice-leak probe ({label} expected): {text!r}", ok, str(gin)))
    return results


# separate from DETERMINISTIC_CHECKS because this returns a LIST of CheckResults (one per
# probe sentence) instead of a single CheckResult - run_all handles both shapes.
PROBE_BATTERIES = [check_advice_leak_probes]


# ---------------------------------------------------------------------------
# Section B - agent-behavior, LLM-dependent, run N times each
# ---------------------------------------------------------------------------

def _tools_called(trace):
    called = []
    for step in trace:
        for act in step.get("actions", []):
            called.append(act["tool"])
    return called


def why_routes_to_context_tool(session_prefix="eval-why-route"):
    """Core routing check: a 'why' question about a US stock should call
    get_price_move_context, not plain get_stock_history. This is the single most important
    check in this file - if routing doesn't work, none of the caveat/hedging behavior below
    matters, because the model never sees any headlines to hedge about."""
    session_id = f"{session_prefix}-{time.time_ns()}"
    result = pipeline.chat(session_id, f"Why did {US_TICKER} drop this week?", channel="eval")
    called = _tools_called(result["trace"])
    passed = "get_price_move_context" in called
    return passed, f"tools_called={called}"


def ph_why_answer_is_hedged(session_prefix="eval-why-ph-hedge"):
    """PH 'why' question: PH now gets real headlines via Google News (same as US), so it should
    be held to the same bar as the US check below - hedged language, not a flat causal claim.
    Same heuristic caveat as the US version: a screening pass, not ground truth."""
    session_id = f"{session_prefix}-{time.time_ns()}"
    result = pipeline.chat(session_id, f"Why did {PH_TICKER} move recently?", channel="eval")
    answer = result["answer"].lower()
    passed = any(w in answer for w in HEDGE_WORDS)
    return passed, result["answer"]


def ph_why_discloses_gap_when_no_news(session_prefix="eval-why-ph-gap"):
    """Conditional check, not a straight pass/rate stochastic test: PH news is real now, so it
    won't be empty every run the way it used to be. This only asserts the 'no news' disclosure
    on the specific runs where the tool genuinely found nothing to ground on - fetched directly
    (deterministic, no LLM) right before the pipeline call so we know which case we're in.
    Runs where news WAS found are reported as skipped rather than counted as a failure, since
    the model correctly has something to reason over in that case."""
    ctx = tools.get_price_move_context(PH_TICKER, *_last_n_days(14))
    if not ctx.get("ok") or ctx.get("news"):
        return None, f"skipped - news was found ({len(ctx.get('news', []))} headlines) or tool errored"
    session_id = f"{session_prefix}-{time.time_ns()}"
    result = pipeline.chat(session_id, f"Why did {PH_TICKER} move recently?", channel="eval")
    answer = result["answer"].lower()
    passed = any(w in answer for w in PH_GAP_WORDS)
    return passed, result["answer"]


def us_why_answer_is_hedged(session_prefix="eval-why-hedge"):
    """US 'why' question: answer should contain hedging language, not a flat causal claim.
    Same heuristic caveat as above - this is a screening pass, real judgment needs a human
    reading the transcript for tone, not just keyword presence."""
    session_id = f"{session_prefix}-{time.time_ns()}"
    result = pipeline.chat(session_id, f"Why did {US_TICKER} drop this week?", channel="eval")
    answer = result["answer"].lower()
    passed = any(w in answer for w in HEDGE_WORDS)
    return passed, result["answer"]


# ---------------------------------------------------------------------------
# Section C - volatility & correlation, deterministic, against REAL live data (not synthetic).
#
# The trick for testing correlation against live data without knowing the "right" answer in
# advance: correlate a ticker with itself. Pearson correlation of any series with itself is
# mathematically guaranteed to be exactly 1.0, regardless of what the actual prices are - so
# this is a hard ground-truth check using real yfinance data, not a soft eyeball check.
# Same idea for the annualized-volatility relationship: daily * sqrt(252) must hold no matter
# what the live number is, since that's the formula, not a market fact.
# ---------------------------------------------------------------------------

def check_volatility_manual_calc() -> CheckResult:
    """The tool's daily_volatility_pct should match an independent stdev calc from its own
    series - same ground-truth pattern as Section A's pct_change check."""
    result = tools.get_volatility(US_TICKER, *_last_n_days(30))
    if not result.get("ok"):
        return CheckResult("volatility matches manual stdev calc", False,
                            f"tool returned error: {result.get('error')}")
    closes = [r["close"] for r in result["series"]]
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    manual_daily = round((variance ** 0.5) * 100, 3)
    ok = abs(manual_daily - result["daily_volatility_pct"]) < 0.01
    return CheckResult("volatility matches manual stdev calc", ok,
                        f"tool={result['daily_volatility_pct']} manual={manual_daily}")


def check_volatility_annualized_relationship() -> CheckResult:
    """annualized_volatility_pct must equal daily_volatility_pct * sqrt(252) - a fixed formula,
    true no matter what the live daily figure happens to be that day."""
    result = tools.get_volatility(US_TICKER, *_last_n_days(30))
    if not result.get("ok"):
        return CheckResult("annualized = daily * sqrt(252)", False,
                            f"tool returned error: {result.get('error')}")
    expected = round(result["daily_volatility_pct"] * (252 ** 0.5), 2)
    ok = abs(expected - result["annualized_volatility_pct"]) < 0.05
    return CheckResult("annualized = daily * sqrt(252)", ok,
                        f"tool={result['annualized_volatility_pct']} expected={expected}")


def check_volatility_bad_ticker_errors_cleanly() -> CheckResult:
    result = tools.get_volatility(BAD_TICKER, *_last_n_days(30))
    ok = result.get("ok") is False and "error" in result
    return CheckResult("volatility: bad ticker errors cleanly", ok, str(result))


def check_volatility_compact_strips_returns_series() -> CheckResult:
    """Same regression tripwire as the why-tool's compact() check: 'returns_series' and
    'series' must be stripped before going back to the model, or every 'why is X volatile'
    answer would blow up the token budget with a full daily-returns dump."""
    result = tools.get_volatility(US_TICKER, *_last_n_days(30))
    compact = tools.compact(result)
    ok = "series" not in compact and "returns_series" not in compact and "daily_volatility_pct" in compact
    return CheckResult("volatility: compact() strips series/returns_series", ok, str(list(compact.keys())))


def check_correlation_self_is_one() -> CheckResult:
    """A ticker correlated with ITSELF must come back as exactly 1.0 - mathematically
    guaranteed regardless of the actual live prices, so this is a real ground-truth check
    without needing to know what TSLA/MSFT correlation 'should' be."""
    result = tools.get_correlation(US_TICKER, US_TICKER, *_last_n_days(30))
    if not result.get("ok"):
        return CheckResult("correlation: ticker vs itself is 1.0", False,
                            f"tool returned error: {result.get('error')}")
    ok = abs(result["correlation"] - 1.0) < 0.0001
    return CheckResult("correlation: ticker vs itself is 1.0", ok, f"correlation={result['correlation']}")


def check_correlation_in_valid_range() -> CheckResult:
    """Pearson correlation is mathematically bounded to [-1, 1] for ANY two series - true for
    any pair of real tickers regardless of what the live number actually is."""
    result = tools.get_correlation(US_TICKER, CORR_TICKER, *_last_n_days(60))
    if not result.get("ok"):
        return CheckResult("correlation: value within [-1, 1]", False,
                            f"tool returned error: {result.get('error')}")
    ok = -1.0001 <= result["correlation"] <= 1.0001
    return CheckResult("correlation: value within [-1, 1]", ok, f"correlation={result['correlation']}")


def check_correlation_bad_ticker_errors_cleanly() -> CheckResult:
    result = tools.get_correlation(US_TICKER, BAD_TICKER, *_last_n_days(30))
    ok = result.get("ok") is False and "error" in result
    return CheckResult("correlation: bad ticker errors cleanly", ok, str(result))


def check_correlation_compact_strips_scatter() -> CheckResult:
    """Same regression tripwire pattern: 'scatter' must be stripped, 'correlation' must not
    be, or the model never sees the actual number it's supposed to report."""
    result = tools.get_correlation(US_TICKER, CORR_TICKER, *_last_n_days(60))
    compact = tools.compact(result)
    ok = "scatter" not in compact and "correlation" in compact
    return CheckResult("correlation: compact() strips scatter, keeps correlation", ok,
                        str(list(compact.keys())))


def check_volatility_stable_vs_growth_sanity() -> CheckResult:
    """Soft sanity check, not a hard fail: a historically stable blue-chip (KO) SHOULD usually
    come back less volatile than a historically choppy growth stock (TSLA) over the same
    window. This can occasionally flip on an unusual week for either stock, so treat a FAIL
    here as 'investigate', not 'the tool is broken' - same caveat as Section A's news-list
    check."""
    growth = tools.get_volatility(US_TICKER, *_last_n_days(60))
    stable = tools.get_volatility(STABLE_TICKER, *_last_n_days(60))
    if not (growth.get("ok") and stable.get("ok")):
        return CheckResult("volatility: growth stock usually swings more than a stable one",
                            False, f"{US_TICKER}={growth.get('error')} {STABLE_TICKER}={stable.get('error')}")
    ok = growth["annualized_volatility_pct"] > stable["annualized_volatility_pct"]
    return CheckResult("volatility: growth stock usually swings more than a stable one", ok,
                        f"{US_TICKER}={growth['annualized_volatility_pct']}% "
                        f"{STABLE_TICKER}={stable['annualized_volatility_pct']}%")


DETERMINISTIC_CHECKS_C = [
    check_volatility_manual_calc,
    check_volatility_annualized_relationship,
    check_volatility_bad_ticker_errors_cleanly,
    check_volatility_compact_strips_returns_series,
    check_correlation_self_is_one,
    check_correlation_in_valid_range,
    check_correlation_bad_ticker_errors_cleanly,
    check_correlation_compact_strips_scatter,
    check_volatility_stable_vs_growth_sanity,
]


STOCHASTIC_CHECKS = [
    ("Why-question routes to get_price_move_context", why_routes_to_context_tool, 5),
    ("PH why-answer contains hedging language", ph_why_answer_is_hedged, 5),
    ("US why-answer contains hedging language", us_why_answer_is_hedged, 5),
    ("PH why-question discloses gap when no news found", ph_why_discloses_gap_when_no_news, 5),
]


def run_stochastic(name, fn, n) -> StochasticResult:
    res = StochasticResult(name=name, n=n, passes=0)
    for i in range(n):
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"EXCEPTION: {e!r}"
        if passed is None:   # conditional check declined to apply this run (e.g. news was found)
            res.skipped += 1
        else:
            res.passes += int(passed)
        res.details.append(detail)
    return res


# ---------------------------------------------------------------------------
def run_all(write_path="eval_why_results.txt"):
    lines = ["=== Section A: deterministic tool checks (run once each) ==="]
    a_pass = 0
    for fn in DETERMINISTIC_CHECKS:
        r = fn()
        a_pass += int(r.passed)
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"[{mark}] {r.name} :: {r.detail}")
    lines.append(f"Section A: {a_pass}/{len(DETERMINISTIC_CHECKS)} passed\n")

    lines.append("=== Section A2: advice-leak probe battery (deterministic, varied phrasings) ===")
    for battery_fn in PROBE_BATTERIES:
        battery_results = battery_fn()
        b_pass = sum(int(r.passed) for r in battery_results)
        for r in battery_results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"[{mark}] {r.name} :: {r.detail}")
        lines.append(f"{battery_fn.__name__}: {b_pass}/{len(battery_results)} passed\n")

    lines.append("=== Section C: volatility & correlation checks (deterministic, real live data) ===")
    c_pass = 0
    for fn in DETERMINISTIC_CHECKS_C:
        r = fn()
        c_pass += int(r.passed)
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"[{mark}] {r.name} :: {r.detail}")
    lines.append(f"Section C: {c_pass}/{len(DETERMINISTIC_CHECKS_C)} passed "
                 f"(last check is a soft sanity check - a FAIL there means 'investigate', not 'broken')\n")

    lines.append("=== Section B: agent-behavior checks (pass RATE over N runs) ===")
    for name, fn, n in STOCHASTIC_CHECKS:
        r = run_stochastic(name, fn, n)
        rate_str = f"{r.rate:.0%}" if r.rate is not None else "n/a"
        skip_note = f", {r.skipped} skipped" if r.skipped else ""
        lines.append(f"{r.name}: {r.passes}/{r.n - r.skipped}{skip_note} ({rate_str})")
        for i, d in enumerate(r.details, 1):
            detail = str(d)
            if "\n" in detail or len(detail) > 100:
                # full text, indented as its own block, so long answers stay readable
                # instead of being squeezed onto one line with everything else
                lines.append(f"    run {i}:")
                for line in detail.splitlines():
                    lines.append(f"        {line}")
            else:
                lines.append(f"    run {i}: {detail}")
    lines.append("")

    out = "\n".join(lines)
    print(out)
    with open(write_path, "w", encoding="utf-8") as f:
        f.write(out)
    return out


if __name__ == "__main__":
    run_all()