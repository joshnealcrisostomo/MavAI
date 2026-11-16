"""Disambiguation module: detect when a question has no clear stock to act on, and ask.

Follows the Week 2 slot-filling idiom: a few-shot classifier extracts the key slot (which
stock), and if that slot can't be filled from the message or the recent conversation, we ask
a natural clarifying question instead of guessing. This is what makes follow-ups like
"what about last month?" work - the ticker carries over from memory.

The actual company-name -> ticker resolution and date parsing happen inside the agent; this
step only decides whether we have enough to proceed at all.
"""
import re
from typing import Optional
from pydantic import BaseModel
from . import entities, schemas

_AMBIGUOUS_REF = re.compile(
    r"\b(it|that stock|the other one|the same one|this one)\b", re.IGNORECASE
)

class SlotCheck(BaseModel):
    """The one slot that matters before we can call a price tool: which stock."""
    has_target: bool                         # is there a stock to act on, here or from context?
    company_or_ticker: Optional[str] = None  # what the user named, if anything
    clarifying_question: Optional[str] = None  # filled only when has_target is False


_SYSTEM = (
    "You decide whether a user's message to a stock-price assistant names a specific stock to "
    "look up, OR clearly refers to one from the conversation so far (e.g. 'it', 'that stock', "
    "'what about last month'). \n"
    "Return ONLY a JSON object: "
    '{"has_target": true/false, "company_or_ticker": "<name or null>", '
    '"clarifying_question": "<question or null>"}.\n'
    "Set has_target true if a company/ticker is named or unambiguously implied by context. "
    "Definition or concept questions about finance terms (e.g. 'what is a moving average', "
    "'explain the P/E ratio') do NOT need a specific stock - set has_target true for those. "
    "Set it false only when the user clearly wants a price/number for a stock but you cannot "
    "tell which one - then write a short, friendly clarifying_question asking which stock."
)

_FEW_SHOT = [
    {"role": "user", "content": "[no prior conversation]\nWhat's the price of Apple?"},
    {"role": "assistant", "content": '{"has_target": true, "company_or_ticker": "Apple", "clarifying_question": null}'},
    {"role": "user", "content": "[no prior conversation]\nWhat were the min and max last quarter?"},
    {"role": "assistant", "content": '{"has_target": false, "company_or_ticker": null, "clarifying_question": "Sure - which stock would you like the min and max for?"}'},
    {"role": "user", "content": "[earlier: user asked about Microsoft (MSFT)]\nwhat about the month before?"},
    {"role": "assistant", "content": '{"has_target": true, "company_or_ticker": "Microsoft", "clarifying_question": null}'},
    {"role": "user", "content": "[no prior conversation]\nShould I buy some tech stocks?"},
    {"role": "assistant", "content": '{"has_target": false, "company_or_ticker": null, "clarifying_question": "Happy to help - which specific stock or ticker are you interested in?"}'},
    {"role": "user", "content": "[no prior conversation]\nwhat's a moving average?"},
    {"role": "assistant", "content": '{"has_target": true, "company_or_ticker": null, "clarifying_question": null}'},
]


def check(user_input: str, history: str = ""):
    """Return (needs_clarification: bool, question_or_none, slotcheck).

    Falls back to proceeding (no clarification) if the classifier output can't be parsed -
    better to attempt an answer than to nag the user on a parse glitch.
    """
    
    # EXCEPTION: if the message also contains a dangling reference
    # and there's no prior conversation to resolve it against, a known-stock mention elsewhere
    # in the message does NOT mean the message is fully resolved
    has_ambiguous_ref = bool(_AMBIGUOUS_REF.search(user_input))

    # EXCEPTION 1: Ambiguous ref with NO history to look back on. If the SAME message also names a
    # known stock (e.g. "What's NVDA trading at and how did it move this week?"), the "it" resolves
    # to that stock - proceed. Only ask when there's a dangling ref and nothing named to bind it to.
    if has_ambiguous_ref and not history.strip():
        if entities.mentions_known_stock(user_input):
            return False, None, None
        q = ("Which stock do you mean? This is the start of our conversation, so I don't have an "
             "earlier stock to go on - could you name the stock you're asking about?")
        return True, q, None
    
    # EXCEPTION 2: Known stock mentioned, AND no ambiguous words -> safe to bypass LLM
    if entities.mentions_known_stock(user_input) and not has_ambiguous_ref:
        return False, None, None

    ctx = f"[earlier conversation:\n{history}\n]" if history else "[no prior conversation]"
    messages = [{"role": "system", "content": _SYSTEM}, *_FEW_SHOT,
                {"role": "user", "content": f"{ctx}\n{user_input}"}]

    inst, err = schemas.parse_with_retry(SlotCheck, messages)
    if inst is None:
        return False, None, None  # safe default: let the agent try
    if not inst.has_target:
        q = inst.clarifying_question or "Which stock are you asking about? (e.g. AAPL, MSFT)"
        return True, q, inst
    return False, None, inst
