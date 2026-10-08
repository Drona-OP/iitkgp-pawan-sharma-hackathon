"""Risk Radar: signals arriving, story clusters with corroboration, the gate log, and the evidence."""

from __future__ import annotations

import statistics

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from seismo.schemas import Signal
from seismo.ui.common import (
    DECISION_COLOUR,
    EVENT_NAMES,
    EVENT_SHORT,
    GLOSSARY,
    INK,
    SEVERITY,
    band,
    base_layout,
    chip,
    entity_label,
    get_controller,
    get_store,
    header,
    headline,
    records,
)


def magnitude_strip(events: list[Signal]) -> go.Figure:
    chrono = sorted(events, key=lambda s: s.as_of)
    span = (chrono[-1].as_of - chrono[0].as_of).total_seconds() if len(chrono) > 1 else 3600
    width_ms = max(60_000.0, span * 1000 / 220)
    fig = go.Figure(go.Bar(
        x=[s.as_of for s in chrono], y=[s.impact_score for s in chrono], width=width_ms,
        marker_color=[SEVERITY[band(s.impact_score)] if s.status == "active" else "#5B6B80" for s in chrono],
        hovertext=[f"<b>{entity_label(s)}</b>, impact {s.impact_score} ({s.status})<br>{EVENT_NAMES[s.event.primary]}"
                   f"<br>{s.corroboration.independent_publishers} publishers, {s.corroboration.social_authors} accounts"
                   f"<br>{headline(s)[:90]}" for s in chrono],
        hoverinfo="text",
    ))
    fig.add_hline(y=7.5, line_dash="dot", line_color=SEVERITY["severe"], line_width=1,
                  annotation_text="Stress-test gate: impact 8 and above", annotation_position="top left",
                  annotation_font_color=SEVERITY["severe"], annotation_font_size=12)
    base_layout(fig, "Event magnitude over time (one bar per story update)", 240)
    fig.update_layout(showlegend=False, bargap=0)
    fig.update_yaxes(range=[0, 10.5], dtick=2, title="Impact")
    return fig


def story_frame(events: list[Signal]) -> pd.DataFrame:
    rows = []
    for s in events:
        c = s.corroboration
        rows.append({
            "Updated (UTC)": s.as_of.strftime("%m-%d %H:%M"),
            "Entity": entity_label(s),
            "Event": EVENT_SHORT[s.event.primary] + (f" / {s.event.subtype.replace('_', ' ').lower()}" if s.event.subtype else ""),
            "Impact": s.impact_score,
            "Sentiment": s.sentiment_score,
            "Publishers": c.independent_publishers,
            "Accounts": c.social_authors,
            "Status": s.status,
            "Flags": ", ".join(f.replace("_", " ") for f in s.flags),
            "Headline": headline(s)[:140],
        })
    return pd.DataFrame(rows)


def gate_frame(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        failed = [c["name"] for c in d["checks"] if not c["passed"]]
        rows.append({
            "Time (UTC)": d["as_of"][5:16].replace("T", " "),
            "Entity": d["entity_id"],
            "Impact": d["impact_score"],
            "Seismo": d["decision"],
            "Naive": d["naive_decision"],
            "Failed checks": ", ".join(failed) if d["decision"] in ("REVIEW", "LOG") and failed else "",
            "Analog": d.get("scenario") or "",
        })
    return pd.DataFrame(rows)


def signal_detail(sig: Signal) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Sentiment (conf. {sig.sentiment_confidence:.2f})", f"{sig.sentiment_score:+.2f}")
    c2.metric(f"Impact ({band(sig.impact_score)})", f"{sig.impact_score} / 10", help=GLOSSARY["Impact"])
    c3.metric("Independent publishers", sig.corroboration.independent_publishers, help=GLOSSARY["Corroboration"])
    c4.metric("Relevance", sig.relevance, help=GLOSSARY["Relevance"])
    sub = f", {sig.event.subtype.replace('_', ' ').lower()}" if sig.event.subtype else ""
    st.markdown(f"**Event:** {EVENT_NAMES[sig.event.primary]}{sub} (confidence {sig.event.confidence:.2f}, {sig.event.method})")
    c = sig.corroboration
    pubs = ", ".join(c.publishers) or "none"
    st.markdown(
        f"**Corroboration:** {c.independent_publishers} independent publisher(s) ({pubs}); "
        f"{c.publishers_60m} in the last 60 minutes; {c.social_authors} social account(s)"
        f"{' - coordinated reposts detected' if c.coordinated else ''}; "
        f"authoritative source: {'yes' if c.authoritative else 'no'}; status: **{sig.status}**"
    )
    if sig.p_large_move is not None:
        st.markdown(f"**Calibrated chance of a two-sigma move:** {sig.p_large_move:.0%}")
    for ev in sig.evidence:
        link = f" ([source]({ev.url}))" if ev.url else ""
        st.markdown(f"<blockquote class='evidence'>{ev.span}</blockquote>", unsafe_allow_html=True)
        st.caption(f"{ev.publisher}{link} - weight {ev.weight:.2f}")
    st.caption("Models: " + ", ".join(f"{k} {v}" for k, v in sig.model_versions.items()))
    if sig.synthetic:
        st.caption("Synthetic replay data: paraphrased reconstruction or fictional scenario, not real news.")


def render() -> None:
    header("Risk Radar", "News, filings and social posts become one evolving signal per story: entity-level sentiment, "
           "event class, impact 1-10, and how many independent voices confirm it.")
    live_view()


@st.fragment(run_every="2s")
def live_view() -> None:
    store = get_store()
    controller = get_controller()
    st.markdown(f"<p class='radar-status'>{'REPLAY RUNNING' if controller.running() else 'IDLE'}"
                f"{' / ' + controller.pack_name if controller.pack_name else ''}</p>", unsafe_allow_html=True)
    events = store.signals(grain="event", limit=800)
    if not events:
        st.info("No signals yet. Pick a scenario in the sidebar and press Start replay "
                "(or run `python -m seismo live` to stream GDELT, SEC EDGAR and Bluesky).")
        return
    stories = store.latest_events(limit=200)
    decisions = records("gate")
    stats = store.stats()
    docs = store.signals(grain="document", limit=500)
    pipeline = [s.latency_ms.get("pipeline", s.latency_ms.get("engine", 0.0)) for s in docs]
    p95 = statistics.quantiles(pipeline, n=20)[-1] if len(pipeline) >= 20 else (max(pipeline) if pipeline else 0.0)
    m = st.columns(6)
    m[0].metric("Documents", f"{stats['documents']:,}")
    m[1].metric("Stories", f"{len(stories):,}")
    m[2].metric("Severe stories", f"{sum(s.impact_score >= 8 for s in stories):,}", help="Impact 8 or more")
    m[3].metric("Auto-triggers", f"{sum(d['decision'] == 'TRIGGER' for d in decisions)}",
                delta=f"naive: {sum(d['naive_decision'] == 'TRIGGER' for d in decisions)}", delta_color="off",
                help="Gate decisions that launched a stress test, versus what a naive impact>7 rule would fire")
    m[4].metric("Engine p50", f"{stats['median_engine_ms']:.1f} ms")
    m[5].metric("Pipeline p95", f"{p95:.1f} ms", help="Document received to signal published (CPU)")

    st.plotly_chart(magnitude_strip(events), width="stretch", config={"displayModeBar": False})

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("**Stories** (one row per event cluster, latest state)")
        st.dataframe(
            story_frame(stories), hide_index=True, width="stretch", height=380,
            column_config={
                "Impact": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%d", width="small", help=GLOSSARY["Impact"]),
                "Sentiment": st.column_config.NumberColumn(format="%+.2f", width="small"),
                "Publishers": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%d", width="small", help=GLOSSARY["Corroboration"]),
                "Accounts": st.column_config.NumberColumn(width="small", help="Distinct social accounts"),
                "Headline": st.column_config.TextColumn(width="large"),
            },
        )
    with right:
        st.markdown("**Trigger gate log** (decision changes only)")
        if decisions:
            frame = gate_frame(decisions[::-1][:60])
            st.dataframe(
                frame.style.map(lambda v: f"color: {DECISION_COLOUR.get(v, INK)}; font-weight: 600", subset=["Seismo", "Naive"]),
                hide_index=True, width="stretch", height=380,
            )
        else:
            st.caption("No gate decisions yet.")

    st.markdown("**Why this score?**")
    by_id = {s.signal_id: s for s in stories[:80]}
    chosen = st.selectbox(
        "Pick a story to see its evidence", options=list(by_id),
        format_func=lambda sid: f"{by_id[sid].as_of:%m-%d %H:%M}  {entity_label(by_id[sid])}  impact {by_id[sid].impact_score}  {headline(by_id[sid])[:70]}",
        key="inspect_story",
    )
    if chosen:
        sig = by_id[chosen]
        flags = "".join(chip(f.replace("_", " ")) for f in sig.flags)
        if flags:
            st.markdown(flags, unsafe_allow_html=True)
        signal_detail(sig)
        gate = [d for d in decisions if d.get("cluster_id") == sig.cluster_id]
        if gate:
            last = gate[-1]
            st.markdown(f"**Gate:** {chip(last['decision'], DECISION_COLOUR.get(last['decision']))}", unsafe_allow_html=True)
            for c in last["checks"]:
                st.markdown(f"{'✅' if c['passed'] else '❌'} **{c['name']}** - {c['detail']}")
