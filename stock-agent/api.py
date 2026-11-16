"""API Endpoint module: FastAPI REST channel over the same core.pipeline as the UI.

  GET  /health        - liveness + reference-doc count
  POST /chat          - one-shot JSON answer (summary, chart spec, tool results, trace)
  POST /chat/stream   - the same answer pushed as SSE (one final event; the pipeline itself
                        is multi-call so we send the result once it's ready)
"""
import json
import os
import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core import pipeline, rag

# root_path only matters when the API is served under a path prefix (the Docker/nginx setup mounts
# it under /api). When run directly on its own port (our server deploy), leave it empty so /docs
# and /openapi.json resolve correctly. Docker sets API_ROOT_PATH=/api.
app = FastAPI(title="Stock Price Agent API", version="1.0",
              root_path=os.getenv("API_ROOT_PATH", ""))


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _run(req: ChatRequest):
    """Run the pipeline and shape a JSON-serialisable payload (no raw chart series bloat)."""
    session_id = req.session_id or f"api-{uuid.uuid4().hex[:8]}"
    result = pipeline.chat(session_id, req.message, channel="api")
    resp = result["response"]
    return {
        "session_id": session_id,
        "answer": result["answer"],
        "tickers": resp.tickers,
        "chart": resp.chart.model_dump(),
        "clarification": result["clarification"],
        "guard": result["guard"],
        "tools_called": [t.get("tool") for t in result["tool_results"] if t.get("ok")],
        "trace": result["trace"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "reference_docs": rag.doc_count()}


@app.post("/chat")
def chat(req: ChatRequest):
    """One-shot JSON answer."""
    return _run(req)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Same answer over SSE. A 'message' event carries the payload, then 'done' closes it."""
    def event_gen():
        payload = _run(req)
        yield {"event": "message", "data": json.dumps(payload)}
        yield {"event": "done", "data": json.dumps({"done": True})}

    return EventSourceResponse(event_gen())
