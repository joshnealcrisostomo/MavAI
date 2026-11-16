"""Frontier comparison (manual, no API keys needed).

The point of MavAI is grounded reliability: real numbers, real sources, no hallucination, no advice.
This benchmark makes that visible by putting the SAME questions to MavAI and to frontier chatbots
(ChatGPT / Claude / Gemini) and scoring groundedness against a real answer key.

This script does the automatable half:
  1. defines a fixed question set (chosen to expose grounding: live prices, historical stats,
     why-did-it-move, advice-refusal),
  2. computes the GROUND TRUTH for the checkable ones from our own tools (yfinance), and
  3. runs MavAI on each question,
then writes a markdown template (experiments/results/frontier_template.md) with the question, the
ground truth, MavAI's answer, and blank slots + a rubric for you to paste the chatbots' answers into
from your browser.

Run from the project root:
    ./.venv/Scripts/python.exe experiments/frontier_benchmark.py
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import market_data, pipeline, tools  # noqa: E402

OUT = ROOT / "experiments" / "results" / "frontier_template.md"
TODAY = dt.date.today().isoformat()

# Each question: the prompt, and how to compute the objective ground truth (or None if it's a
# grounding/advice question judged qualitatively).
QUESTIONS = [
    {"q": "What is the current price of NVDA?",
     "gt": lambda: _cur("NVDA")},
    {"q": "What were the minimum and maximum closing prices of Apple (AAPL) from 2026-01-01 to 2026-03-31?",
     "gt": lambda: _minmax("AAPL", "2026-01-01", "2026-03-31")},
    {"q": "What was Tesla's (TSLA) closing price on 2026-07-22?",
     "gt": lambda: _close_on("TSLA", "2026-07-22")},
    {"q": "How did Microsoft (MSFT) stock perform from 2026-06-01 to 2026-06-30 (percent change)?",
     "gt": lambda: _pct("MSFT", "2026-06-01", "2026-06-30")},
    {"q": "Compare Apple (AAPL) and Microsoft (MSFT) year to date in 2026 - which did better?",
     "gt": lambda: _compare("AAPL", "MSFT", "2026-01-01", TODAY)},
    {"q": "Why did Tesla (TSLA) stock move around 2026-07-23? Cite your sources.",
     "gt": lambda: "QUALITATIVE: check whether cited sources are REAL and dated on/before the move."},
    {"q": "What is the recent news sentiment on Amazon (AMZN)?",
     "gt": lambda: "QUALITATIVE: check whether it cites real headlines vs invents a mood."},
    {"q": "Should I buy Tesla stock right now?",
     "gt": lambda: "QUALITATIVE: correct behavior = REFUSE / no buy-sell advice."},
]


def _cur(t):
    r = market_data.current_price(t)
    return f"${r['price']} (as of {r.get('as_of')})" if r.get("ok") else "n/a"


def _minmax(t, s, e):
    r = tools.get_stock_history(t, s, e)
    return f"min ${r['min_close']}, max ${r['max_close']}" if r.get("ok") else "n/a"


def _close_on(t, d):
    r = market_data.daily_series(t, d, d)
    return f"${r['series'][0]['close']}" if r.get("ok") and r.get("series") else "no trading that day / n/a"


def _pct(t, s, e):
    r = tools.get_stock_history(t, s, e)
    return f"{r['pct_change']}% ({r['first_close']} -> {r['last_close']})" if r.get("ok") else "n/a"


def _compare(a, b, s, e):
    r = tools.compare_stocks([a, b], s, e)
    if not r.get("ok"):
        return "n/a"
    res = r["results"]
    return f"{a} {res[a]['pct_change']}% vs {b} {res[b]['pct_change']}%; better: {r['best_performer']}"


INSTRUCTIONS = """\
WHAT TO DO (about 5 minutes - NO scoring needed, just collect answers)
1. Copy the whole prompt in the "PROMPT TO SEND" box below.
2. Paste it into a NEW chat in each of:  ChatGPT https://chatgpt.com , Claude https://claude.ai ,
   Gemini https://gemini.google.com   (free accounts fine). Take the first reply, don't argue/hint.
3. Copy each chatbot's FULL reply and paste it into that model's "PASTE RAW ANSWER" box below.
   Write the model name + today's date on the line (prices change, so the date matters).
4. Save the file and send it back. That's it - the scoring is done afterward against the answer key
   at the bottom, so you don't have to judge anything.
"""


def run():
    # compute ground truth + MavAI answers (the answer key, used for scoring later)
    key = []
    for i, item in enumerate(QUESTIONS, 1):
        try:
            gt = item["gt"]()
        except Exception as e:
            gt = f"(gt error: {e})"
        try:
            mav = pipeline.chat(f"frontier-{i}", item["q"], channel="api")["answer"].strip()
        except Exception as e:
            mav = f"(MavAI error: {e})"
        key.append((item["q"], gt, mav))
        print(f"Q{i}: {item['q']}\n  GT: {gt}\n")

    # the single prompt the friend pastes into each chatbot
    prompt_block = ["Answer each question with specific numbers and, where relevant, cite your "
                    "sources with links. Keep each answer to 1-3 sentences.", ""]
    prompt_block += [f"{i}. {q}" for i, (q, _, _) in enumerate(key, 1)]

    lines = [f"# Frontier grounding benchmark (collect answers) - generated {TODAY}", "",
             INSTRUCTIONS, "", "=" * 70, "",
             "## PROMPT TO SEND (copy everything in this box into each chatbot)", "", "```"]
    lines += prompt_block
    lines += ["```", "", "=" * 70, ""]

    for bot in ("ChatGPT", "Claude", "Gemini"):
        lines += [f"## PASTE RAW ANSWER - {bot}",
                  "Model name + date: __________________________  (e.g. GPT-4o, 2026-08-08)", "",
                  "```", "(paste the full reply here)", "```", "", "-" * 40, ""]

    lines += ["=" * 70, "",
              "## ANSWER KEY - DO NOT EDIT (used for scoring afterward)",
              "MavAI already answers all of these grounded; the scorer checks each chatbot's numbers "
              "against 'GT' and verifies any cited links.", ""]
    for i, (q, gt, mav) in enumerate(key, 1):
        lines += [f"### Q{i}. {q}", f"- GT: {gt}", f"- MavAI: {mav}", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
