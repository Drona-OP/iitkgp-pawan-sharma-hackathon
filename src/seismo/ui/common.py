"""Shared dashboard pieces: theme, colours, store access, the replay controller, small helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from seismo.config import load_settings
from seismo.schemas import EventClass, Signal
from seismo.store.sqlite import SQLiteStore

INK = "#E6EDF5"
MUTED = "#8A9BB0"
GRID = "#24324A"
PANEL = "#172234"
# One colour meaning everywhere, colour-blind safe: red negative, green/teal positive, amber watch.
NEG = "#E2574C"
POS = "#3FB68B"
WATCH = "#E0B04C"
ACCENT = "#4FA3A5"
SEVERITY = {"calm": "#4FA3A5", "watch": "#E0B04C", "serious": "#E8873A", "severe": "#E2574C"}
DECISION_COLOUR = {"TRIGGER": NEG, "REVIEW": WATCH, "LOG": MUTED, "RETRACT": ACCENT}

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
EVENT_SHORT = {
    EventClass.MACROECONOMIC: "Macro", EventClass.GEOPOLITICAL: "Geopolitical",
    EventClass.CREDIT_EVENT: "Credit", EventClass.MA_CORPORATE_ACTION: "M&A, corporate",
    EventClass.PRODUCT_STRATEGY: "Product", EventClass.EARNINGS_GUIDANCE: "Earnings",
    EventClass.LEGAL_REGULATORY: "Legal", EventClass.OPERATIONAL_ESG: "Operational",
    EventClass.MANAGEMENT_GOVERNANCE: "Governance", EventClass.OTHER: "Other",
}
MACRO_LABELS = {
    "MACRO:FED": "Fed", "MACRO:ECB": "ECB", "MACRO:RBI": "RBI", "MACRO:OPEC": "OPEC+",
    "MACRO:OIL": "Crude oil", "MACRO:RATES": "UST yields", "MACRO:MARKET": "US market",
}

# Tooltip definitions: every risk term on screen is explained where it appears.
GLOSSARY = {
    "Impact": "Severity on 1-10. With the calibrated model, impact 8 means roughly a 70-80% chance of a two-sigma price move.",
    "Novelty": "RavenPack-style: 100 for the first story on an entity in 24 hours, 100/(1+n) for the n-th similar one.",
    "Relevance": "0-100: how central the entity is to the story (title, first mention, mention count).",
    "Corroboration": "Independent owners of news and filings reporting the story. Sister outlets, syndicated copies, lookalike domains and coordinated reposts never add voices.",
    "ECL": "Expected credit loss (IFRS 9 / RBI ECL): Stage 1 = 12-month PD x LGD x EAD; Stage 2 = lifetime; Stage 3 = LGD x EAD.",
    "CET1": "Common Equity Tier 1 capital over risk-weighted assets. Floors: Basel 7.0% (4.5% + 2.5% buffer), RBI 8.0% (5.5% + 2.5%).",
    "Reverse stress": "The severity multiple of the analog at which CET1 first falls to the floor.",
    "EAD": "Exposure at default = drawn + 75% credit conversion x undrawn.",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp {{ font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1480px; }}
h1.radar-title {{ font-weight: 600; font-size: 1.95rem; letter-spacing: -0.015em; margin: 0; color: {INK}; }}
p.radar-sub {{ color: {MUTED}; margin: 0.15rem 0 1.0rem 0; font-size: 0.97rem; max-width: 80ch; }}
p.radar-status {{ color: {MUTED}; font-size: 0.85rem; margin: 0 0 0.4rem 0; font-family: 'IBM Plex Mono', monospace; }}
div[data-testid="stMetricValue"] {{ font-weight: 500; }}
blockquote.evidence {{
  border-left: 3px solid {ACCENT}; margin: 0.4rem 0 0.2rem 0; padding: 0.3rem 0.9rem;
  color: {INK}; background: rgba(79, 163, 165, 0.08); border-radius: 0 6px 6px 0;
}}
span.chip {{ display: inline-block; padding: 0.05rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
  margin-right: 0.3rem; border: 1px solid {GRID}; color: {INK}; background: {PANEL}; }}
pre.memo {{ white-space: pre-wrap; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
  background: {PANEL}; border: 1px solid {GRID}; border-radius: 8px; padding: 0.9rem 1rem; color: {INK}; }}
</style>
"""

settings = load_settings()


def header(title: str, subtitle: str) -> None:
    st.markdown(f"<h1 class='radar-title'>{title}</h1><p class='radar-sub'>{subtitle}</p>", unsafe_allow_html=True)


def chip(text: str, colour: str | None = None) -> str:
    style = f" style='border-color:{colour};color:{colour}'" if colour else ""
    return f"<span class='chip'{style}>{text}</span>"


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


def base_layout(fig: go.Figure, title: str | None = None, height: int = 300) -> go.Figure:
    fig.update_layout(
        title={"text": title or "", "x": 0, "font": {"size": 15, "color": INK}},
        height=height, margin={"l": 8, "r": 8, "t": 40 if title else 10, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans, sans-serif", "color": MUTED, "size": 12},
        legend={"orientation": "h", "y": -0.15, "font": {"color": INK}},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_resource
def get_store() -> SQLiteStore:
    return SQLiteStore(settings.path("store.path"))


def records(kind: str, limit: int = 5000) -> list[dict]:
    return [json.loads(r) for r in get_store().records(kind, limit=limit)]


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
            self.proc.wait(timeout=5)  # type: ignore[union-attr]


@st.cache_resource
def get_controller() -> ReplayController:
    return ReplayController(settings.root)


def packs() -> list[Path]:
    order = ["deepseek_2025", "svb_2023", "tariff_2025", "red_team", "quiet_day", "demo_synthetic"]
    found = sorted((settings.root / "data" / "replay").glob("*.jsonl"))
    return sorted(found, key=lambda p: (order.index(p.stem) if p.stem in order else 99, p.stem))


def sidebar_replay() -> None:
    controller = get_controller()
    with st.sidebar:
        st.subheader("Replay a scenario")
        options = packs()
        if options:
            default = settings.path("replay.default_pack")
            index = options.index(default) if default in options else 0
            pack = st.selectbox("Scenario pack", options, index=index, format_func=lambda p: p.stem.replace("_", " "))
            speed_label = st.select_slider("Clock speed", options=["60x", "600x", "3600x", "36000x", "Instant"], value="3600x")
            speed = 0.0 if speed_label == "Instant" else float(speed_label.rstrip("x"))
            left, right = st.columns(2)
            if left.button("Start replay", type="primary", width="stretch"):
                controller.start(pack, speed)
            if right.button("Stop", width="stretch"):
                controller.stop()
        status = "replay running" if controller.running() else "idle"
        st.caption(f"Status: {status}" + (f" ({controller.pack_name})" if controller.pack_name else ""))
        st.divider()
        st.caption("Live sources: `python -m seismo live` streams GDELT, SEC EDGAR and Bluesky into the same store.")
        st.caption("API: http://localhost:8000/docs")
