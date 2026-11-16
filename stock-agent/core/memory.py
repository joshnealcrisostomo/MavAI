"""Memory 

ConversationSummaryBufferMemory + per-session registry
    - This is how the agent "remembers" what you were just talking about

Idea:
    - Keep the last few turns word-for-word so recent context stays sharp, but once the buffer
    fills up, fold the older turns into a running summary instead of just deleting them. 
    - With that, the conversation can go on forever without blowing up the token budget

Why does this matter for a stock agent, specifically? 
    - If you ask "what about last month?" without naming a ticker, the agent needs to know you were 
    just talking about AAPL, or whatever, two messages ago. 
    - The summary is what carries that forward - it's basically the agent's short-term memory of 
    "which stock are we even discussing right now"

The per-session registry at the bottom just makes sure each user/session gets their own
separate memory, so nobody's conversation history bleeds into someone else's.
"""
from . import config, llm


class ConversationSummaryBufferMemory:
    def __init__(self, buffer_window: int = None):
        self.buffer_window = buffer_window or config.BUFFER_WINDOW
        self.buffer = []      # Stores raw message history (best for single-session chat)
        self.summary = ""     # Periodic LLM summarisation (capacity > detail)

    def _summarise(self, older, prior):
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in older)
        prompt = (
            "Update the running summary of a stock-price conversation. Keep it factual, under "
            "80 words. Preserve which tickers/companies and date ranges were discussed so later "
            "follow-ups can be resolved.\n\n"
            f"Existing summary:\n{prior or '(none)'}\n\n"
            f"New messages:\n{convo}\n\nUpdated summary:"
        )
        
        # low temperature to keep the summary deterministic/factual and not creative
        return llm.complete(
            [{"role": "system", "content": "You write concise running summaries."},
             {"role": "user", "content": prompt}],
            temperature=0.0,
        )

    # Instead of just discarding the oldest turn when the buffer is full, it folds the evicted turn into the running summary before dropping it
    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        # buffer full? drop the oldest and fold it into the summary
        while len(self.buffer) > self.buffer_window:
            # FIFO, ensures info isn't lost, just compressed
            overflow = self.buffer.pop(0)
            self.summary = self._summarise([overflow], self.summary)

    # Summary block first, then raw recent turns 
    def context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Summary of earlier conversation]\n{self.summary}")
        if self.buffer:
            recent = "\n".join(f"{m['role']}: {m['content']}" for m in self.buffer)
            parts.append(f"[Recent turns]\n{recent}")
        # both fed to the LLM as context
        return "\n\n".join(parts)


# One memory per session id, shared across both channels in this process
_sessions: dict[str, ConversationSummaryBufferMemory] = {}

# Each session_id (like each browser tab/user) gets its own isolated memory object
# Ensures User A's ticker history never leaks into User B's context
def get(session_id: str) -> ConversationSummaryBufferMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationSummaryBufferMemory()
    return _sessions[session_id]
