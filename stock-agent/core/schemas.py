"""Structured Outputs module: Pydantic schemas + the validate-and-retry helper.

Follows the Week 2 idiom - the model emits JSON, we pull it out with a regex, validate it
against a Pydantic model, and on a validation error we feed the exact error message back and
let the model try again (up to 3 times) before giving up.

Two schemas live here:
  StockQuery    - the structured plan the disambiguation/agent step fills in (slot-filling target)
  StockResponse - the typed answer envelope both the UI and the API consume downstream
"""
import datetime as dt
import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from . import llm


# the parsed user request
class StockQuery(BaseModel):
    """What the user is asking for, normalised. Filled by the disambiguation/planning step."""
    intent: Literal["history_stats", "current_price", "moving_average",
                    "compare", "concept", "unknown"] = "unknown"
    tickers: list[str] = Field(default_factory=list, description="resolved US tickers, upper-case")
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    window: Optional[int] = Field(default=None, description="moving-average window in trading days")

    @field_validator("tickers", mode="before")
    @classmethod
    def _upper(cls, v):
        if isinstance(v, str):
            v = [v]
        return [str(t).upper().strip() for t in (v or [])]

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _parse_date(cls, v):
        if v in (None, "", "null"):
            return None
        if isinstance(v, dt.date):
            return v
        return dt.date.fromisoformat(str(v).strip())

    @model_validator(mode="after")
    def _range_sane(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError(f"end_date {self.end_date} is before start_date {self.start_date}")
        return self


# the answer the UI / API render
class ChartSpec(BaseModel):
    """How the UI should chart the data. The model picks the kind; data comes from the tools."""
    kind: Literal["line", "candlestick", "moving_average", "comparison",
                  "volatility", "correlation", "none"] = "none"
    title: str = ""


class StockResponse(BaseModel):
    """Typed answer envelope. The summary is LLM-written; numbers/series are attached in code
    from the tool results so displayed figures are never hallucinated."""
    summary: str
    tickers: list[str] = Field(default_factory=list)
    chart: ChartSpec = Field(default_factory=ChartSpec)
    disclaimer: str = ""

    @field_validator("tickers", mode="before")
    @classmethod
    def _upper(cls, v):
        if isinstance(v, str):
            v = [v]
        return [str(t).upper().strip() for t in (v or [])]


class ForecastVerdict(BaseModel):
    """The agent's judgment of a price-only forecast in light of the news available before it.
    This is a critique of an ALGORITHM's reliability - NOT a buy/sell recommendation or a price
    prediction. Produced via parse_with_retry over a vet_forecast bundle (see core/vetting.py)."""
    stance: Literal["trust", "caution", "reject"]
    rationale: str
    cited_news: list[str] = Field(default_factory=list,
                                  description="URLs taken ONLY from the bundle's news_before list")
    confidence: Literal["low", "medium", "high"] = "low"


# the Week 2 retry helper
def extract_json(text: str) -> str:
    """Pull the first {...} block out of a model reply (it sometimes wraps JSON in prose)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return match.group()


def parse_with_retry(model_cls, messages, model=None, max_tries=3, temperature=0.0):
    """Ask the model for JSON, validate against model_cls, retry feeding back the error.

    Returns (instance, None) on success or (None, error_string) after max_tries failures.
    """
    convo = list(messages)
    last_err = ""
    for attempt in range(max_tries):
        raw = llm.complete(convo, model=model, temperature=temperature)
        try:
            data = json.loads(extract_json(raw))
            return model_cls.model_validate(data), None
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = str(e)
            # hand the model its own bad output plus the exact error, ask it to fix
            convo = list(messages) + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    f"That did not validate. Error:\n{last_err}\n"
                    "Return ONLY a corrected JSON object, no prose."},
            ]
    return None, last_err