"""LLMOps logging: one JSON line per request to stdout + one MLflow run per request.

Each request logs latency_ms, prompt_tokens, completion_tokens, total_tokens, model,
tool calls made, and an estimated cost_usd from the per-model price table in config.
"""
import json
import sys
import time

from . import config

# mlflow-skinny gives us the tracking client without the heavy server deps
try:
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    _MLFLOW = True
except Exception:  # logging must never take the request down
    _MLFLOW = False


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
    return round(
        prompt_tokens / 1_000_000 * price["input"]
        + completion_tokens / 1_000_000 * price["output"],
        8,
    )


class RequestLogger:
    """Use as a context manager around one chat request, then call .finish()."""

    def __init__(self, channel: str, model: str = None):
        self.channel = channel
        self.model = model or config.AGENT_MODEL
        # token counts accumulate across the whole ReAct loop, not just one call
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def add_usage(self, usage):
        """Fold one model call's token usage into the running total."""
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def finish(self, **extra):
        latency_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        pt, ct = self.prompt_tokens, self.completion_tokens
        tt = pt + ct
        record = {
            "channel": self.channel,
            "model": self.model,
            "latency_ms": latency_ms,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "cost_usd": estimate_cost(self.model, pt, ct),
            **extra,
        }
        # 1) the required single JSON line
        sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()
        # 2) the same numbers into MLflow
        if _MLFLOW:
            try:
                with mlflow.start_run(run_name=f"{self.channel}-request"):
                    mlflow.log_params({"model": self.model, "channel": self.channel})
                    mlflow.log_metrics({
                        "latency_ms": latency_ms,
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt,
                        "cost_usd": record["cost_usd"],
                    })
            except Exception:
                pass
        self.record = record
        return record

    def __exit__(self, *exc):
        # if the caller didn't call finish() (e.g. an error path), still log latency
        if not hasattr(self, "record"):
            self.finish(error=repr(exc[1]) if exc[1] else None)
        return False
