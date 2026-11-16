"""ReAct Agent module: a reasoning + acting loop over the yfinance tools.

The loop is the standard tool-calling pattern:
  1. send the question (plus tool schemas) to the 70B model
  2. if the model asks for a tool, run it and feed the result back
  3. repeat until the model stops calling tools and writes a final answer

We record a thought/action/observation trace at every step so the UI can show the agent's
reasoning - that visibility is both a demo asset and what lets the team explain the loop.
Displayed numbers come from the tool results (ground truth), never from the model's prose.
"""
import datetime as dt
import json

from . import config, llm, prompts, tools

# default prompt when run_agent is called without one. Single source of truth is prompts.RULES,
# which also covers US vs PH tool routing. {today} is filled at call time.
DEFAULT_SYSTEM_PROMPT = prompts.RULES

# Captures the Thought and Action parts of the ReAct trace
def _trace_step(message):
    """Build a trace entry from a model message that requested tool(s)."""
    return {
        "thought": (message.content or "").strip(),  # 70B often narrates before calling
        "actions": [{"tool": tc.function.name, "args": tc.function.arguments}
                    for tc in (message.tool_calls or [])],
    }


def run_agent(user_input: str, history: str = "", system_prompt: str = None,
              max_steps: int = None, vet_mode: str = "investor"):
    """Run the ReAct loop. Returns a dict:

        answer       - the model's final natural-language reply (str)
        tool_results - full tool outputs incl. chart series, in call order (list[dict])
        trace        - per-step thought/action/observation for the UI (list[dict])
        usages       - token usage from each model call, for the LLMOps log (list)
    """
    max_steps = max_steps or config.MAX_AGENT_STEPS
    system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).format(
        today=dt.date.today().isoformat())
    # append the user-facing vetting sensitivity toggle (investor/analyst) - only shifts how
    # cautious the "how reliable is the forecast" answer is; appended after .format so its
    # braces never hit the format call
    system_prompt += "\n\n" + prompts.vet_sensitivity_clause(vet_mode)

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.append({"role": "system", "content": f"Conversation so far:\n{history}"})
    messages.append({"role": "user", "content": user_input})

    tool_results, trace, usages = [], [], []

    # ReAct Loop
    # Each iteration is one Thought/Action/Observation cycle
    # Bounded by max_steps so the agent can't loop forever
    for _ in range(max_steps):
        try:
            msg, usage = llm.chat_with_tools(messages, tools.TOOL_SPECS)
        except Exception as e:
            # tool-calling failed even after retries - degrade gracefully, never crash the UI
            return {"answer": "Sorry, I had trouble working that out. Could you rephrase the "
                              "question, maybe naming the stock and date range explicitly?",
                    "tool_results": tool_results, "trace": trace, "usages": usages,
                    "error": str(e)}
        usages.append(usage)

        # Final Answer
        if not msg.tool_calls:
            # no tool wanted -> this is the final answer
            return {"answer": (msg.content or "").strip(),
                    "tool_results": tool_results, "trace": trace, "usages": usages}

        # record the reasoning + the tool calls the model wants, then run them
        step = _trace_step(msg)
        # the assistant turn that requested the tools must go back into the convo verbatim
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls],
        })

        # model sees the Observation and decides its next Thought/Action
        observations = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # run each requested tool
            result = tools.run_tool(tc.function.name, args)
            tool_results.append(result)
            
            # strip the heavy chart data
            compact = tools.compact(result)
            observations.append({"tool": tc.function.name, "ok": result.get("ok"),
                                 "result": compact})
            # feed the (compact) result back to the model
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(compact)})
        step["observations"] = observations
        trace.append(step)

    # ran out of steps - summarise what we have rather than loop forever
    return {"answer": "I gathered some data but couldn't fully resolve the request within the "
                      "step limit. Please try narrowing the question.",
            "tool_results": tool_results, "trace": trace, "usages": usages}
