"""The one pipeline both channels (Streamlit + FastAPI) call.

Flow per turn:
  input guardrails -> disambiguation (maybe ask) -> RAG reference -> ReAct agent (tools)
  -> build typed StockResponse + chart spec -> output guardrail (disclaimer) -> save memory
  -> LLMOps log

Returns a dict the UI/API render. We do not stream here because the ReAct loop is multi-call;
the final answer is returned whole, which keeps tool results and chart data consistent.
"""
from . import (agent, config, disambiguation, guardrails, memory,
               observability, prompts, rag)
from .schemas import ChartSpec, StockResponse


def _infer_chart(tool_results, user_input):
    """Pick a chart kind, tickers, and the source tool result from what the agent called.

    Returns (ChartSpec, tickers, chart_data) where chart_data is the tool result dict the UI
    charts from (or None).
    """
    want_candle = "candlestick" in user_input.lower() or "candle" in user_input.lower()
    # walk results in reverse so the most recent/relevant tool wins
    for r in reversed(tool_results):
        if not r.get("ok"):
            continue
        tool = r.get("tool")
        if tool == "compare_stocks":
            return (ChartSpec(kind="comparison",
                              title=f"Performance comparison ({r.get('start_date')} to {r.get('end_date')})"),
                    r.get("tickers", []), r)
        if tool == "get_price_move_context":
            return (ChartSpec(kind="line",
                              title=f"{r.get('ticker')} price ({r.get('start_date')} to {r.get('end_date')})"),
                    [r.get("ticker")], r)
        if tool == "get_volatility":
            return (ChartSpec(kind="volatility",
                              title=f"{r.get('ticker')} daily returns ({r.get('start_date')} to {r.get('end_date')})"),
                    [r.get("ticker")], r)
        if tool == "get_correlation":
            return (ChartSpec(kind="correlation",
                              title=f"{r.get('ticker_a')} vs {r.get('ticker_b')} daily returns "
                                    f"(correlation {r.get('correlation')})"),
                    [r.get("ticker_a"), r.get("ticker_b")], r)
        if tool == "get_moving_average":
            return (ChartSpec(kind="moving_average",
                              title=f"{r.get('ticker')} {r.get('window')}-day moving average"),
                    [r.get("ticker")], r)
        if tool == "get_stock_history":
            # candlestick needs OHLC, which only US (yfinance) data has; PH is close-only
            kind = "candlestick" if (want_candle and r.get("has_ohlc")) else "line"
            return (ChartSpec(kind=kind,
                              title=f"{r.get('ticker')} closing price ({r.get('start_date')} to {r.get('end_date')})"),
                    [r.get("ticker")], r)
        if tool == "get_current_price":
            # PH current price has no series to chart; US current price carries a 5-day series
            if not r.get("series"):
                return ChartSpec(kind="none", title=""), [r.get("ticker")], None
            return (ChartSpec(kind="line", title=f"{r.get('ticker')} recent price"),
                    [r.get("ticker")], r)
    return ChartSpec(kind="none", title=""), [], None


def _sources_block(tool_results):
    """Build a clickable markdown 'Sources' list from the news that grounded a 'why did it move'
    answer - only the headlines dated on/before the move (before_move=true). Links come straight
    from the tool result, so they're the real URLs (the model never has to retype them)."""
    seen, lines = set(), []
    for r in tool_results:
        if not r.get("ok") or r.get("tool") != "get_price_move_context":
            continue
        for n in r.get("news", []):
            link, title = n.get("link"), n.get("title")
            if n.get("before_move") and link and title and link not in seen:
                seen.add(link)
                pub = f" - {n.get('publisher')}" if n.get("publisher") else ""
                lines.append(f"- [{title}]({link}){pub}")
    return "\n\n**Sources:**\n" + "\n".join(lines[:5]) if lines else ""


def chat(session_id: str, user_input: str, channel: str = "api",
         prompt_variant: str = None, vet_mode: str = "investor") -> dict:
    """Run one turn. Returns:
        answer        - final text (with disclaimer)
        response      - StockResponse (typed, what the API serialises)
        tool_results  - full tool outputs incl. chart series (for the UI charts)
        trace         - ReAct steps for the UI
        guard         - which guard fired, or "none"
        clarification - True if we asked a clarifying question instead of answering
    """
    log = observability.RequestLogger(channel=channel)
    with log:
        mem = memory.get(session_id)
        history = mem.context()

        # 1) input guardrails - block before spending any tokens on a bad request
        gin = guardrails.check_input(user_input)
        if not gin["allowed"]:
            mem.add("user", user_input)
            mem.add("assistant", gin["reply"])
            log.finish(guard=gin["guard"], clarification=False)
            resp = StockResponse(summary=gin["reply"])
            return {"answer": gin["reply"], "response": resp, "tool_results": [],
                    "trace": [], "guard": gin["guard"], "clarification": False}

        # 2) disambiguation - if we can't tell which stock, ask instead of guessing
        needs, question, _ = disambiguation.check(user_input, history)
        if needs:
            mem.add("user", user_input)
            mem.add("assistant", question)
            log.finish(guard="none", clarification=True)
            resp = StockResponse(summary=question)
            return {"answer": question, "response": resp, "tool_results": [],
                    "trace": [], "guard": "none", "clarification": True}

        # 3) RAG reference - ground company/sector and finance-term answers
        reference = rag.retrieve(user_input)
        agent_history = history
        if reference:
            agent_history = (history + "\n\n" if history else "") + f"REFERENCE:\n{reference}"

        # 4) ReAct agent - reason + call tools
        variant = prompts.VARIANTS.get(prompt_variant or prompts.DEFAULT)
        result = agent.run_agent(user_input, history=agent_history, system_prompt=variant,
                                 vet_mode=vet_mode)
        for u in result["usages"]:
            log.add_usage(u)

        # if the agent degraded (tool-calling failed after retries), return the friendly message
        # but DON'T save it to memory - a "sorry" turn would pollute the context and make the
        # next follow-up ("it"/"that stock") unresolvable.
        if result.get("error"):
            log.finish(guard="none", clarification=False, error="agent_degraded")
            resp = StockResponse(summary=result["answer"])
            return {"answer": result["answer"], "response": resp, "tool_results": [],
                    "trace": result["trace"], "guard": "none", "clarification": False,
                    "chart_data": None}

        # 5) typed response + chart spec (numbers/series stay grounded in the tool results)
        chart, tickers, chart_data = _infer_chart(result["tool_results"], user_input)
        answer = guardrails.ensure_disclaimer(result["answer"] + _sources_block(result["tool_results"]))
        resp = StockResponse(summary=result["answer"], tickers=tickers, chart=chart,
                             disclaimer=guardrails.DISCLAIMER.strip())

        # 6) memory + metrics
        mem.add("user", user_input)
        mem.add("assistant", result["answer"])
        log.finish(guard="none", clarification=False,
                   tools_called=[t.get("tool") for t in result["tool_results"]],
                   steps=len(result["trace"]))

        return {"answer": answer, "response": resp, "tool_results": result["tool_results"],
                "trace": result["trace"], "guard": "none", "clarification": False,
                "chart_data": chart_data}