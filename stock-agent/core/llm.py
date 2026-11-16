"""Thin wrapper over the OpenAI SDK pointed at Groq. One place to call the model.

Supports a pool of API keys: when one key hits its rate limit (HTTP 429), we rotate to the next
and retry, so several free Groq keys together give a lot more headroom. The current key index is
sticky across calls, so once we move off an exhausted key we stay on the next good one.

Three entry points:
  complete()        - one-shot text, used by the classifiers and summariser (8b model)
  stream_completion - token-by-token streaming (not used in the main pipeline path)
  chat_with_tools() - a single tool-calling round-trip, used by the ReAct loop (70b model)
"""
import sys

from openai import OpenAI, RateLimitError

from . import config

# the key pool. a key can't be empty or the SDK errors at construction.
_keys = config.LLM_API_KEYS or ["no-key-set"]
_idx = 0
_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=_keys[_idx])


def _rotate():
    """Move to the next key in the pool and rebuild the client. Returns the new index."""
    global _idx, _client
    _idx = (_idx + 1) % len(_keys)
    _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=_keys[_idx])
    sys.stderr.write(f"[llm] rate limited, rotating to key #{_idx + 1}/{len(_keys)}\n")
    sys.stderr.flush()
    return _idx


def complete(messages, model=None, temperature=0.2):
    """One-shot completion. Defaults to the fast/cheap model. Rotates keys on rate limit."""
    last_err = None
    for _ in range(len(_keys)):
        try:
            resp = _client.chat.completions.create(
                model=model or config.FAST_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            last_err = e
            _rotate()
    raise last_err  # every key was rate limited


def stream_completion(messages, model=None, temperature=0.2):
    """Stream the answer token by token. Yields (delta_text, usage); usage arrives on the final
    chunk. Rotates keys on rate limit before the stream starts."""
    stream = last_err = None
    for _ in range(len(_keys)):
        try:
            stream = _client.chat.completions.create(
                model=model or config.FAST_MODEL,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            break
        except RateLimitError as e:
            last_err = e
            _rotate()
    if stream is None:
        raise last_err
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta, None
        if usage is not None:
            yield "", usage


def chat_with_tools(messages, tools, model=None, temperature=0.0, max_retries=3):
    """One tool-calling round-trip for the ReAct agent. Returns (message, usage).

    Handles two failure modes:
      - rate limit (429): rotate to the next key and retry (does not use a tool-retry attempt)
      - Groq's intermittent malformed tool call (tool_use_failed): retry with a small
        temperature bump to nudge the model back into valid format
    """
    last_err = None
    tool_attempt = 0
    keys_tried = 1  # current key counts as tried
    while tool_attempt <= max_retries:
        try:
            resp = _client.chat.completions.create(
                model=model or config.AGENT_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature + 0.3 * tool_attempt,
            )
            return resp.choices[0].message, resp.usage
        except RateLimitError as e:
            last_err = e
            if keys_tried < len(_keys):
                _rotate()
                keys_tried += 1
                continue  # same tool_attempt, fresh key
            raise  # all keys rate limited
        except Exception as e:
            last_err = e
            if "tool_use_failed" in str(e) or "Failed to call a function" in str(e):
                tool_attempt += 1
                continue  # transient tool-format hiccup, try again
            raise
    raise last_err
