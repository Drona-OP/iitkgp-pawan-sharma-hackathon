"""Try It: a judge types a headline and gets a full signal; the red-team button replays the fake."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from seismo.schemas import Document, SourceType, stable_id
from seismo.ui.common import (
    DECISION_COLOUR,
    EVENT_NAMES,
    GLOSSARY,
    INK,
    MUTED,
    band,
    base_layout,
    chip,
    header,
    settings,
)

EXAMPLES = [
    "Tesla recalls 200,000 vehicles over a braking software fault",
    "JPMorgan beats profit estimates as trading revenue jumps 20%",
    "Regulators close a regional lender after a run on deposits; FDIC named receiver",
    "Apple pie recipes for the long weekend",
    "Nvidia shares slide 9% premarket as a rival unveils a cheaper AI model",
    "Infosys gains after a large deal win while TCS slides on weak guidance",
    "Short seller accuses Adani Group of stock manipulation and accounting fraud",
    "HDFC Bank shares rise as deposit growth beats estimates; SBI flat",
]


@st.cache_resource
def engine_factory():
    from seismo.engine import build_engine

    return build_engine(settings)


def fresh_engine():
    from seismo.engine import Engine

    base = engine_factory()
    return Engine(base.universe, base.backend, settings, base.impact)


def analyze_box() -> None:
    st.markdown("**Type a headline (and optionally a body).** Scored by the same engine as the replays.")
    pick = st.selectbox("Or start from an example", ["(type your own)"] + EXAMPLES)
    c1, c2 = st.columns([3, 1])
    title = c1.text_input("Headline", value="" if pick == "(type your own)" else pick)
    source = c2.selectbox("Source type", ["news", "social", "filing"])
    body = st.text_area("Body (optional)", height=80)
    publisher = st.text_input("Publisher domain", value="wire-one.example" if source == "news" else "bsky.app",
                              help="Try a lookalike such as wire-one-alerts.example to see the lookalike flag.")
    if not st.button("Score it", type="primary") or not (title or body):
        return
    now = datetime.now(UTC)
    doc = Document(doc_id=stable_id("d", "ui", title, body, now.isoformat()), source="ui",
                   source_type=SourceType(source), publisher=publisher, author="did:plc:you" if source == "social" else None,
                   published_at=now, title=title, body=body)
    engine = fresh_engine()
    result = engine.process_full(doc)
    if not result.doc_signals:
        st.info("No tracked entity found, and the text is not macro or geopolitical news, so no signal is emitted. "
                "That is the linker refusing to guess (for example 'Apple pie' does not link to AAPL).")
        return
    from seismo.scenarios import build_gate

    gate = build_gate(settings)
    for sig, ev in zip(result.doc_signals, result.event_signals + [None] * len(result.doc_signals), strict=False):
        st.markdown(f"### {sig.entity.name} ({sig.entity.id})")
        cols = st.columns(5)
        cols[0].metric("Sentiment", f"{sig.sentiment_score:+.2f}")
        cols[1].metric(f"Impact ({band(sig.impact_score)})", f"{sig.impact_score}/10", help=GLOSSARY["Impact"])
        cols[2].metric("Event", EVENT_NAMES[sig.event.primary].split(" ")[0], help=f"{sig.event.primary.value}/{sig.event.subtype}")
        cols[3].metric("Relevance", sig.relevance, help=GLOSSARY["Relevance"])
        cols[4].metric("Engine time", f"{sig.latency_ms.get('engine', 0):.1f} ms")
        if sig.flags:
            st.markdown("".join(chip(f.replace("_", " ")) for f in sig.flags), unsafe_allow_html=True)
        st.markdown(f"<blockquote class='evidence'>{sig.evidence[0].span}</blockquote>", unsafe_allow_html=True)
        if ev is not None:
            d = gate.evaluate(ev, sig)
            st.markdown(f"Gate on this single report: {chip(d.decision, DECISION_COLOUR.get(d.decision))} "
                        f"naive rule: {chip(d.naive_decision, DECISION_COLOUR.get(d.naive_decision))}", unsafe_allow_html=True)
            for c in d.checks:
                st.markdown(f"{'✅' if c.passed else '❌'} **{c.name}** - {c.detail}")
        with st.expander("Raw signal JSON (schema v1.1)"):
            st.json(sig.model_dump(mode="json"))


def red_team() -> None:
    st.markdown("**Red team: a fake bank-run headline from a lookalike account, 40 coordinated reposts, then an "
                "official denial** (fictional bank; the same shape as the May 2023 fake Pentagon explosion).")
    if not st.button("Run the red-team scenario"):
        return
    from seismo.ingest.replay import load_pack
    from seismo.scenarios import build_gate
    from seismo.signals.gate import GateConsumer

    engine = fresh_engine()
    gate = GateConsumer(build_gate(settings))
    timeline = []
    for doc in load_pack(settings.root / "data" / "replay" / "red_team.jsonl"):
        result = engine.process_full(doc)
        for sig in result.doc_signals:
            gate.on_signal(sig)
        for sig in result.event_signals:
            d = gate.gate.evaluate(sig, None)
            gate.on_signal(sig)
            timeline.append({"time": doc.published_at, "impact": sig.impact_score, "seismo": d.decision,
                             "naive": "TRIGGER" if (gate._last_doc.get(sig.cluster_id) and abs(gate._last_doc[sig.cluster_id].sentiment_score) * 10 > 7) else "LOG",
                             "accounts": sig.corroboration.social_authors, "publishers": sig.corroboration.publishers_60m,
                             "status": sig.status})
    df = pd.DataFrame(timeline)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["impact"], mode="lines+markers", name="Seismo impact",
                             line={"color": INK, "width": 2}, marker={"color": [DECISION_COLOUR[d] for d in df["seismo"]], "size": 9}))
    naive_fire = df[df["naive"] == "TRIGGER"]
    fig.add_trace(go.Scatter(x=naive_fire["time"], y=[10.3] * len(naive_fire), mode="markers", name="Naive pipeline fires a stress test",
                             marker={"symbol": "x", "size": 12, "color": DECISION_COLOUR["TRIGGER"]}))
    fig.add_hline(y=7.5, line_dash="dot", line_color=MUTED)
    base_layout(fig, "Impact rises with the swarm; the gate holds (amber = REVIEW), then the denial retracts (teal)", 320)
    fig.update_yaxes(range=[0, 11], title="Impact")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    seismo_triggers = int((df["seismo"] == "TRIGGER").sum())
    c = st.columns(4)
    c[0].metric("Naive stress tests fired", int((df["naive"] == "TRIGGER").sum()))
    c[1].metric("Seismo auto-triggers", seismo_triggers)
    c[2].metric("Peak impact while in review", int(df[df["seismo"] == "REVIEW"]["impact"].max()) if (df["seismo"] == "REVIEW").any() else 0)
    c[3].metric("Final status", df["status"].iloc[-1])
    last_review = next((d for d in reversed(gate.log) if d.decision == "REVIEW"), None)
    if last_review:
        st.markdown("Why the gate held:")
        for ch in last_review.checks:
            st.markdown(f"{'✅' if ch.passed else '❌'} **{ch.name}** - {ch.detail}")


def render() -> None:
    header("Try It", "Score any headline with the live engine, or run the red-team attack the gate is built to stop.")
    tab1, tab2 = st.tabs(["Score a headline", "Red team"])
    with tab1:
        analyze_box()
    with tab2:
        red_team()
