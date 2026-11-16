"""Chat UI module: Streamlit conversational interface with Plotly charts.

Shares the exact core.pipeline the FastAPI channel uses, so both answer identically. Each
answer can come with a chart (line / candlestick / moving-average / comparison) built from the
agent's tool results, and an expander showing the ReAct reasoning trace.
"""
import uuid

import plotly.graph_objects as go
import streamlit as st

from core import pipeline, rag

st.set_page_config(page_title="MavAI Stock Price Agent", page_icon="📈", layout="centered")


def build_chart(spec, data):
    """Turn a chart spec + the source tool result into a Plotly figure (or None)."""
    if not data or spec.kind == "none":
        return None
    fig = go.Figure()

    if spec.kind == "candlestick":
        s = data.get("series", [])
        fig.add_trace(go.Candlestick(
            x=[r["date"] for r in s], open=[r["open"] for r in s], high=[r["high"] for r in s],
            low=[r["low"] for r in s], close=[r["close"] for r in s], name=data.get("ticker", "")))
        fig.update_layout(xaxis_rangeslider_visible=False)

    elif spec.kind == "moving_average":
        price = data.get("series", [])
        ma = data.get("ma_series", [])
        if price:
            fig.add_trace(go.Scatter(x=[r["date"] for r in price], y=[r["close"] for r in price],
                                     mode="lines", name="Close", line=dict(color="#9aa0a6")))
        fig.add_trace(go.Scatter(x=[r["date"] for r in ma], y=[r["ma"] for r in ma],
                                 mode="lines", name=f"{data.get('window')}-day MA",
                                 line=dict(color="#1a73e8", width=2)))

    elif spec.kind == "comparison":
        for ticker, series in data.get("comparison_series", {}).items():
            fig.add_trace(go.Scatter(x=[r["date"] for r in series], y=[r["pct"] for r in series],
                                     mode="lines", name=ticker))
        fig.update_layout(yaxis_title="% change from start")

    elif spec.kind == "volatility":
        rs = data.get("returns_series", [])
        colors = ["#34a853" if r["return_pct"] >= 0 else "#ea4335" for r in rs]
        fig.add_trace(go.Bar(x=[r["date"] for r in rs], y=[r["return_pct"] for r in rs],
                             marker_color=colors, name="Daily return"))
        fig.update_layout(yaxis_title="Daily return (%)")

    elif spec.kind == "correlation":
        sc = data.get("scatter", [])
        fig.add_trace(go.Scatter(
            x=[p["return_a_pct"] for p in sc], y=[p["return_b_pct"] for p in sc],
            mode="markers", marker=dict(color="#1a73e8", size=7, opacity=0.7), name="Daily returns"))
        tickers = data.get("ticker_a", ""), data.get("ticker_b", "")
        fig.update_layout(xaxis_title=f"{tickers[0]} daily return (%)",
                          yaxis_title=f"{tickers[1]} daily return (%)")

    else:  # line
        s = data.get("series", [])
        fig.add_trace(go.Scatter(x=[r["date"] for r in s], y=[r["close"] for r in s],
                                 mode="lines", name="Close", line=dict(color="#1a73e8", width=2)))

    fig.update_layout(title=spec.title, height=380, margin=dict(l=10, r=10, t=40, b=10),
                      template="plotly_white")
    return fig


def render_trace(trace):
    """Show the agent's thought/action/observation steps in an expander."""
    if not trace:
        return
    with st.expander(f"🧠 Agent reasoning ({len(trace)} step{'s' if len(trace) != 1 else ''})"):
        for i, step in enumerate(trace, 1):
            if step.get("thought"):
                st.markdown(f"**Step {i} - thought:** {step['thought']}")
            for act in step.get("actions", []):
                st.markdown(f"&nbsp;&nbsp;↳ **calls** `{act['tool']}` with `{act['args']}`")
            for obs in step.get("observations", []):
                ok = "✅" if obs.get("ok") else "⚠️"
                st.markdown(f"&nbsp;&nbsp;{ok} **observes** `{obs['tool']}` → {obs['result']}")


# session state: stable id (isolated memory) + the rendered message log
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued" not in st.session_state:
    st.session_state.queued = None

st.title("📈 MavAI Stock Price Agent")
st.caption("Ask about US and Philippine (PSE) stocks in plain language: prices, history, "
           "comparisons, volatility, and why a stock moved. Its core feature VETS how reliable a "
           "price forecast is for US FAANG / Big-Tech - using news the forecaster could not see - and "
           "flags when to distrust it. A ReAct agent routes to the right tools with memory, "
           "guardrails, and RAG. It never predicts a price or gives buy/sell advice.")

with st.sidebar:
    st.subheader("Try asking")
    examples = [
        "How reliable is the forecast for Tesla right now?",
        "Why did Tesla drop this week?",
        "What were the min and max prices of Apple in Q1 2025?",
        "Compare Apple, Microsoft and Google year to date",
        "Show me Tesla's 20-day moving average for the last 3 months",
        "What's the current price of Globe (GLO)?",
        "How volatile has Jollibee been this year?",
        "Are Apple and Microsoft correlated?",
        "What is a moving average?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.queued = ex
    st.divider()
    st.subheader("Vetting sensitivity")
    vet_label = st.radio(
        "How aggressively should it flag risks to a forecast?",
        ["Investor (precise)", "Analyst (wide net)"],
        index=0, label_visibility="collapsed",
        help="Investor: only flags genuine threats, few false alarms. "
             "Analyst: flags any material catalyst and leans cautious. "
             "Only affects 'how reliable is the forecast' answers.")
    vet_mode = "analyst" if vet_label.startswith("Analyst") else "investor"
    st.divider()
    st.caption(f"Reference docs indexed: **{rag.doc_count()}**")
    st.caption("Models: `llama-3.3-70b` (agent) + `llama-3.1-8b-instant` (classifiers) via Groq")
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.divider()
    with st.expander("⚠️ Known limitations"):
        st.markdown(
            "- Detects and flags catalysts; it does NOT predict prices - no model we tested "
            "(incl. a SOTA foundation model) beats a coin flip on direction\n"
            "- Forecast-vetting covers US FAANG / Big-Tech only\n"
            "- Modest catch-rate; news-retrieval recall is low - its main bottleneck\n"
            "- Daily data only, so intraday catalysts are not captured live\n"
            "- No candlesticks for PH stocks (phisix is close-only, no intraday OHLC)\n"
            "- No real-time prices (daily closes, may be delayed) or fundamentals (EPS, P/E, revenue)\n"
            "- Not financial advice - refuses buy/sell/prediction requests"
        )

# replay the conversation so far
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fig") is not None:
            st.plotly_chart(m["fig"], use_container_width=True, key=f"chart_{i}")
        render_trace(m.get("trace"))

# a new turn: either typed input or a clicked example
prompt = st.chat_input("Ask about a stock...")
if st.session_state.queued and not prompt:
    prompt = st.session_state.queued
    st.session_state.queued = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.chat(st.session_state.session_id, prompt, channel="streamlit",
                                   vet_mode=vet_mode)
        st.markdown(result["answer"])
        fig = build_chart(result["response"].chart, result.get("chart_data"))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_live_{len(st.session_state.messages)}")
        render_trace(result["trace"])

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"],
        "fig": fig, "trace": result["trace"]})