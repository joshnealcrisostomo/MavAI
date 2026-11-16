"""Forecast-vetting judge: turn a vet_forecast bundle into a structured ForecastVerdict.

Reuses the Week-2 structured-output idiom (schemas.parse_with_retry): the model emits JSON, we
validate it against the ForecastVerdict schema and retry on failure. Kept separate from the live
ReAct agent so the experiment harness can score verdicts deterministically. The prompt (VET_RULES)
frames the task as auditing an ALGORITHM's reliability, never as price advice.
"""
from . import config, prompts, schemas


def _bundle_brief(bundle: dict) -> str:
    """Compact, model-facing summary of a vet_forecast bundle: the forecast, the baseline, and the
    pre-event headlines (the only URLs the verdict may cite)."""
    fc, base = bundle.get("forecast", {}), bundle.get("baseline", {})
    lines = [
        f"Ticker: {bundle['ticker']}   as-of date: {bundle['as_of_date']}   "
        f"horizon: {bundle['horizon']} trading days",
        f"Forecasting algorithm ({fc.get('model')}): predicted direction={fc.get('direction')}, "
        f"last_close={fc.get('last_close')}",
        f"Naive-drift baseline direction: {base.get('direction')}",
        "",
        "Pre-event headlines (news_before) - you may cite ONLY these URLs:",
    ]
    news = bundle.get("news_before", [])
    if not news:
        lines.append("  (no relevant pre-event news found)")
    for n in news:
        lines.append(f"  - [{n.get('seendate')}] {n.get('title')}  ({n.get('url')})")
    return "\n".join(lines)


def verdict_messages(bundle: dict, mode: str = None) -> list[dict]:
    user = (
        _bundle_brief(bundle)
        + "\n\nReturn ONLY a JSON object with keys: stance (trust|caution|reject), "
          "rationale (string), cited_news (list of URLs drawn only from news_before), "
          "confidence (low|medium|high)."
    )
    return [{"role": "system", "content": prompts.vet_rules(mode or config.VET_MODE)},
            {"role": "user", "content": user}]


def judge(bundle: dict, mode: str = None):
    """Return (ForecastVerdict, None) or (None, error_string). Does not raise.

    mode selects the auditor by user: 'investor' (default, precise, fewer false alarms) or 'analyst'
    (wide net, flags any material catalyst). Falls back to config.VET_MODE when not given.

    Grounding is enforced in code, not just asked for in the prompt: any cited URL that isn't in the
    bundle's news_before is dropped, so a fabricated/outside source can never reach the user."""
    if not bundle.get("ok"):
        return None, bundle.get("error", "bad bundle")
    verdict, err = schemas.parse_with_retry(schemas.ForecastVerdict, verdict_messages(bundle, mode))
    if verdict is not None:
        allowed = {n.get("url") for n in bundle.get("news_before", []) if n.get("url")}
        verdict.cited_news = [u for u in verdict.cited_news if u in allowed]
    return verdict, err
