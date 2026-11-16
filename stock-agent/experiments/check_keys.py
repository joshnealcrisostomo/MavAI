"""Check remaining rate-limit quota per Groq key, from the response headers.

Groq returns x-ratelimit-* headers on every call (remaining requests/tokens, their limits, and
reset times). This pings each key in the pool with a tiny 1-token request and prints a
"% remaining" for the day, so you can see which keys are fresh and which are near their limit.

Note: this spends ONE request per key (negligible). Run from the project root:
    ./.venv/Scripts/python.exe experiments/check_keys.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

from openai import AuthenticationError, OpenAI, RateLimitError  # noqa: E402

from core import config  # noqa: E402


def mask(k):
    return f"{k[:6]}...{k[-4:]}" if len(k) >= 10 else "????"


def pct(remaining, limit):
    try:
        return f"{float(remaining) / float(limit) * 100:.0f}% left"
    except (TypeError, ValueError, ZeroDivisionError):
        return "?"


def check_key(key, model):
    """Ping one key (against the given model) and read its rate-limit headers. Returns a small
    status dict. Rate limits in Groq are tracked per model, so a key that looks fresh on one
    model can still be exhausted on another - each model needs its own check."""
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=key)
    try:
        # with_raw_response gives us the HTTP headers, not just the parsed body
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        h = raw.headers
        rr, lr = h.get("x-ratelimit-remaining-requests"), h.get("x-ratelimit-limit-requests")
        rt, lt = h.get("x-ratelimit-remaining-tokens"), h.get("x-ratelimit-limit-tokens")
        return {
            "status": "OK",
            "requests": f"{rr}/{lr}  ({pct(rr, lr)})  resets in {h.get('x-ratelimit-reset-requests')}",
            "tokens": f"{rt}/{lt}  ({pct(rt, lt)})  resets in {h.get('x-ratelimit-reset-tokens')}",
        }
    except RateLimitError as e:
        ra = getattr(getattr(e, "response", None), "headers", {}) or {}
        return {"status": "EXHAUSTED (rate limited)", "retry_after": ra.get("retry-after")}
    except AuthenticationError:
        return {"status": "INVALID KEY"}
    except Exception as e:
        return {"status": f"ERROR {type(e).__name__}: {str(e)[:60]}"}


def main():
    keys = config.LLM_API_KEYS
    # AGENT_MODEL is the 70B model eval_ragas.py uses as its judge (RAGAS_JUDGE_MODEL defaults to
    # it) - that's the one that actually hits TPD caps during a RAGAS run, so check it too.
    models = [("fast", config.FAST_MODEL), ("judge (70B)", config.AGENT_MODEL)]
    print(f"Checking {len(keys)} key(s) against {config.LLM_BASE_URL}")
    print(f"(models: {', '.join(f'{label}={m}' for label, m in models)}; "
          f"each key spends 1 request per model)\n")
    for i, k in enumerate(keys, 1):
        print(f"#{i}  {mask(k)}")
        for label, model in models:
            r = check_key(k, model)
            print(f"    [{label}] {model}  ->  {r['status']}")
            if r["status"] == "OK":
                print(f"        requests: {r['requests']}")
                print(f"        tokens:   {r['tokens']}")
            elif r["status"].startswith("EXHAUSTED"):
                print(f"        retry after: {r.get('retry_after')}")
        print()


if __name__ == "__main__":
    main()