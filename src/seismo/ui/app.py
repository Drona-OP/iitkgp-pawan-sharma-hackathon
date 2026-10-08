"""Seismo Risk Radar: signals as they arrive, how severe they are, and why.

Run with `python -m seismo ui` (or `python -m seismo demo` to start a replay as well).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from seismo.config import load_settings  # noqa: E402
from seismo.schemas import EventClass, Signal  # noqa: E402
from seismo.store.sqlite import SQLiteStore  # noqa: E402

INK = "#E6EDF5"
MUTED = "#8A9BB0"
GRID = "#24324A"
SEVERITY = {"calm": "#4FA3A5", "watch": "#E0B04C", "serious": "#E8873A", "severe": "#E2574C"}

EVENT_NAMES = {
    EventClass.MACROECONOMIC: "Macroeconomic",
    EventClass.GEOPOLITICAL: "Geopolitical",
    EventClass.CREDIT_EVENT: "Credit event",
    EventClass.MA_CORPORATE_ACTION: "M&A and corporate action",
    EventClass.PRODUCT_STRATEGY: "Product and strategy",
    EventClass.EARNINGS_GUIDANCE: "Earnings and guidance",
    EventClass.LEGAL_REGULATORY: "Legal and regulatory",
    EventClass.OPERATIONAL_ESG: "Operational and ESG",
    EventClass.MANAGEMENT_GOVERNANCE: "Management and governance",
    EventClass.OTHER: "Other",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 1.4rem; max-width: 1440px; }}
h1.radar-title {{ font-weight: 600; font-size: 2.05rem; letter-spacing: -0.015em; margin: 0; color: {INK}; }}
p.radar-sub {{ color: {MUTED}; margin: 0.15rem 0 1.1rem 0; font-size: 0.98rem; max-width: 72ch; }}
p.radar-status {{ color: {MUTED}; font-size: 0.85rem; margin: 0 0 0.4rem 0; }}
div[data-testid="stMetricValue"] {{ font-weight: 500; }}
blockquote.evidence {{
  border-left: 3px solid {SEVERITY['calm']}; margin: 0.4rem 0 0.2rem 0; padding: 0.3rem 0.9rem;
  color: {INK}; background: rgba(79, 163, 165, 0.08); border-radius: 0 6px 6px 0;
}}
</style>
"""


MACRO_LABELS = {
    "MACRO:FED": "Fed", "MACRO:ECB": "ECB", "MACRO:RBI": "RBI", "MACRO:OPEC": "OPEC+",
    "MACRO:OIL": "Crude oil", "MACRO:RATES": "UST yields", "MACRO:MARKET": "US market",
}
EVENT_SHORT = {
    EventClass.MACROECONOMIC: "Macro", EventClass.GEOPOLITICAL: "Geopolitical",
    EventClass.CREDIT_EVENT: "Credit", EventClass.MA_CORPORATE_ACTION: "M&A, corporate",
    EventClass.PRODUCT_STRATEGY: "Product", EventClass.EARNINGS_GUIDANCE: "Earnings",
    EventClass.LEGAL_REGULATORY: "Legal", EventClass.OPERATIONAL_ESG: "Operational",
    EventClass.MANAGEMENT_GOVERNANCE: "Governance", EventClass.OTHER: "Other",
}


def band(impact: int) -> str:
    if impact >= 8:
        return "severe"
    if impact == 7:
        return "serious"
    if impact >= 4:
        return "watch"
    return "calm"


def entity_label(sig: Signal) -> str:
    return sig.entity.id if sig.entity.type == "company" else MACRO_LABELS.get(sig.entity.id, sig.entity.name)


def headline(sig: Signal) -> str:
    if not sig.evidence:
        return ""
    ev = sig.evidence[0]
    return (ev.title or ev.span).strip()


st.set_page_config(page_title="Seismo Risk Radar", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
settings = load_settings()


@st.cache_resource
def get_store() -> SQLiteStore:
    return SQLiteStore(settings.path("store.path"))


class ReplayController:
    """Runs one replay at a time as a child process that writes into the shared store."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.autostarted = False
        self.pack_name = ""

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, pack: Path, speed: float) -> None:
        self.stop()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "seismo", "replay", "--pack", str(pack), "--speed", str(speed), "--reset"],
            cwd=self.root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.pack_name = pack.stem.replace("_", " ")

    def stop(self) -> None:
        if self.running():
            self.proc.terminate()  # type: ignore[union-attr]


@st.cache_resource
def get_controller() -> ReplayController:
    return ReplayController(settings.root)


store = get_store()
controller = get_controller()
packs = sorted((settings.root / "data" / "replay").glob("*.jsonl"))
default_pack = settings.path("replay.default_pack")

if os.environ.get("SEISMO_AUTOSTART") == "1" and not controller.autostarted and default_pack.exists():
    controller.autostarted = True
    controller.start(default_pack, float(settings.get("replay.speed", 600)))

with st.sidebar:
    st.subheader("Replay a scenario")
    if packs:
        index = packs.index(default_pack) if default_pack in packs else 0
        pack = st.selectbox("Scenario pack", packs, index=index, format_func=lambda p: p.stem.replace("_", " "))
        speed_label = st.select_slider(
            "Clock speed", options=["60x", "300x", "600x", "1800x", "3600x", "Instant"], value="600x"
        )
        speed = 0.0 if speed_label == "Instant" else float(speed_label.rstrip("x"))
        left, right = st.columns(2)
        if left.button("Start replay", type="primary", width="stretch"):
            controller.start(pack, speed)
        if right.button("Stop", width="stretch"):
            controller.stop()
    else:
        st.info("No packs in data/replay. Record one with `python -m seismo live --record <file>`.")
    st.divider()
    st.caption(
        "Live sources: run `python -m seismo live` in a terminal. "
        "Signals appear here as soon as they are scored."
    )

st.markdown(
    "<h1 class='radar-title'>Risk Radar</h1>"
    "<p class='radar-sub'>Signals from news, filings and social posts, scored per company "
    "and per event. Impact runs from 1 (routine) to 10 (severe).</p>",
    unsafe_allow_html=True,
)


def magnitude_strip(signals: list[Signal]) -> go.Figure:
    strongest: dict = {}
    for s in signals:
        best = strongest.get(s.as_of)
        if best is None or s.impact_score > best.impact_score:
            strongest[s.as_of] = s
    chrono = sorted(strongest.values(), key=lambda s: s.as_of)
    span = (chrono[-1].as_of - chrono[0].as_of).total_seconds() if len(chrono) > 1 else 3600
    width_ms = max(60_000.0, span * 1000 / 220)
    fig = go.Figure(
        go.Bar(
            x=[s.as_of for s in chrono],
            y=[s.impact_score for s in chrono],
            width=width_ms,
            marker_color=[SEVERITY[band(s.impact_score)] for s in chrono],
            hovertext=[
                f"<b>{entity_label(s)}</b>, impact {s.impact_score}<br>"
                f"{EVENT_NAMES[s.event.primary]}<br>{headline(s)[:90]}"
                for s in chrono
            ],
            hoverinfo="text",
        )
    )
    fig.add_hline(
        y=7.5, line_dash="dot", line_color=SEVERITY["severe"], line_width=1,
        annotation_text="Stress-test trigger: impact 8 and above", annotation_position="top left",
        annotation_font_color=SEVERITY["severe"], annotation_font_size=12,
    )
    fig.update_layout(
        title={"text": "Event magnitude over time", "x": 0, "font": {"size": 15, "color": INK}},
        height=240, margin={"l": 8, "r": 8, "t": 40, "b": 8}, showlegend=False, bargap=0,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans, sans-serif", "color": MUTED, "size": 12},
        yaxis={"range": [0, 10.5], "dtick": 2, "gridcolor": GRID, "title": "Impact", "zeroline": False},
        xaxis={"gridcolor": GRID, "showgrid": False},
    )
    return fig


def feed_frame(signals: list[Signal]) -> pd.DataFrame:
    rows = []
    for s in signals:
        ev = s.evidence[0] if s.evidence else None
        rows.append(
            {
                "UTC": s.as_of.strftime("%H:%M"),
                "Entity": entity_label(s),
                "Impact": s.impact_score,
                "Sentiment": s.sentiment_score,
                "Event": EVENT_SHORT[s.event.primary],
                "Headline": headline(s)[:160],
                "Source": s.corroboration.source_types[0].value if s.corroboration.source_types else "",
                "Publisher": ev.publisher if ev else "",
            }
        )
    return pd.DataFrame(rows)


def entity_frame(signals: list[Signal]) -> pd.DataFrame:
    rows = [
        {
            "Entity": entity_label(s),
            "Index": s.sentiment_score,
            "Peak impact": s.impact_score,
            "Publishers": s.corroboration.independent_publishers,
        }
        for s in signals
    ]
    return pd.DataFrame(rows)


def signal_detail(sig: Signal) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Sentiment (confidence {sig.sentiment_confidence:.2f})", f"{sig.sentiment_score:+.2f}")
    c2.metric(f"Impact ({band(sig.impact_score)})", f"{sig.impact_score} / 10")
    c3.metric("Novelty", f"{sig.novelty}")
    c4.metric("Relevance", f"{sig.relevance}")
    subtype = f", {sig.event.subtype.replace('_', ' ').lower()}" if sig.event.subtype else ""
    st.markdown(
        f"**Event:** {EVENT_NAMES[sig.event.primary]}{subtype} "
        f"(confidence {sig.event.confidence:.2f}, method {sig.event.method})"
    )
    corr = sig.corroboration
    st.markdown(
        f"**Corroboration:** {corr.independent_publishers} independent publisher(s); "
        f"sources: {', '.join(t.value for t in corr.source_types) or 'none'}; "
        f"authoritative source: {'yes' if corr.authoritative else 'no'}"
    )
    for ev in sig.evidence:
        link = f" ([open source]({ev.url}))" if ev.url else ""
        st.markdown(f"<blockquote class='evidence'>{ev.span}</blockquote>", unsafe_allow_html=True)
        st.caption(f"{ev.publisher}{link}")
    st.caption("Models: " + ", ".join(f"{k} {v}" for k, v in sig.model_versions.items()))
    if sig.synthetic:
        st.caption("This signal comes from synthetic test data, not real news.")


@st.fragment(run_every="2s")
def live_view() -> None:
    status = "Replay running" if controller.running() else "Idle"
    if controller.pack_name:
        status += f", pack: {controller.pack_name}"
    st.markdown(f"<p class='radar-status'>{status}</p>", unsafe_allow_html=True)

    doc_signals = store.signals(grain="document", limit=500)
    if not doc_signals:
        st.info(
            "No signals yet. Start a replay from the sidebar, or run `python -m seismo live` "
            "in a terminal to stream GDELT, SEC EDGAR and Bluesky."
        )
        return

    stats = store.stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents scored", f"{stats['documents']:,}")
    m2.metric("Signals emitted", f"{stats['signals']:,}")
    m3.metric("Severe events (impact 8+)", f"{stats['severe']:,}")
    m4.metric("Median engine time", f"{stats['median_engine_ms']:.1f} ms")

    st.plotly_chart(magnitude_strip(doc_signals), width="stretch", config={"displayModeBar": False})

    left, right = st.columns([13, 7], gap="medium")
    with left:
        st.markdown("**Incoming signals**")
        st.dataframe(
            feed_frame(doc_signals[:80]),
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "UTC": st.column_config.TextColumn(width="small"),
                "Entity": st.column_config.TextColumn(width="small"),
                "Impact": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%d", width="small"),
                "Sentiment": st.column_config.NumberColumn(format="%+.2f", width="small"),
                "Event": st.column_config.TextColumn(width="small"),
                "Headline": st.column_config.TextColumn(width="large"),
            },
        )
    with right:
        st.markdown("**Entity board**")
        entities = store.latest_entities()
        if entities:
            st.dataframe(
                entity_frame(entities),
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "Entity": st.column_config.TextColumn(width="small"),
                    "Index": st.column_config.NumberColumn(format="%+.2f", width="small", help="Decayed, weighted sentiment, -1 to +1"),
                    "Peak impact": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%d", width="small", help="Highest impact in the last 6 hours"),
                    "Publishers": st.column_config.NumberColumn(width="small", help="Independent publishers reporting the peak event"),
                },
            )

    st.markdown("**Why this score?**")
    recent = doc_signals[:60]
    by_id = {s.signal_id: s for s in recent}
    chosen = st.selectbox(
        "Pick a signal to see its evidence",
        options=list(by_id),
        format_func=lambda sid: (
            f"{by_id[sid].as_of:%H:%M}  {entity_label(by_id[sid])}  impact {by_id[sid].impact_score}  "
            f"{headline(by_id[sid])[:70]}"
        ),
        key="inspect_signal",
    )
    if chosen:
        signal_detail(by_id[chosen])


live_view()
