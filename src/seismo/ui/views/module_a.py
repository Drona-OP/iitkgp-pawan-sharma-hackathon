"""Module A page: index weights over time, active weights, the "why" drawer, equity curves, controls."""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from seismo.ui.common import (
    ACCENT,
    MUTED,
    NEG,
    POS,
    WATCH,
    base_layout,
    get_store,
    header,
    records,
    settings,
)

PALETTE = ["#4FA3A5", "#7A9CC6", "#E0B04C", "#C77DBA", "#6CC4A4", "#E8873A", "#9AA9BC", "#5B8DEF",
           "#D96C6C", "#A3C46C", "#F2C46D", "#8E7CC3", "#4DB6AC", "#F48FB1", "#90A4AE", "#FFB74D",
           "#81C784", "#64B5F6", "#BA68C8", "#A1887F"]


INDICES = {"US": "US: 20 S&P 100 names", "IN": "India: 16 Nifty 50 names"}


def short(ticker: str) -> str:
    return ticker.removesuffix(".NS")


def recompute(kappa: float, strict: bool, half_life: float, market: str = "US") -> list[dict]:
    """Re-run Module A over the signals already in the store with different controls."""
    from seismo.module_a.consumer import ModuleAConsumer
    from seismo.scenarios import build_gate
    from seismo.signals.gate import GateConsumer

    module = ModuleAConsumer.from_settings(settings, store=None, strict=strict, market=market)
    module.cfg = replace(module.cfg, kappa=kappa, half_life_hours=half_life)
    module.lam = math.log(2) / (half_life * 3600)
    gate = GateConsumer(build_gate(settings), sinks=[module.on_decision])
    for _, sig in get_store().signals_after(0, limit=100_000):
        if sig.grain == "entity":
            continue
        gate.on_signal(sig)
        module.on_signal(sig)
    return module.snapshots


def render() -> None:
    header("Module A - Tactical index rebalancer",
           "A risk overlay on a 20-stock S&P 100 index or a 16-stock Nifty 50 index: filtered, decayed entity "
           "sentiment tilts capped market-cap weights, scaled by inverse volatility, inside name and sector caps, "
           "with a circuit breaker for corroborated credit, operational and fraud events.")
    us, india = records("weights"), records("weights_in")
    latest_in = india[-1]["as_of"] if india else ""
    latest_us = us[-1]["as_of"] if us else ""
    default = 1 if india and (not us or latest_in >= latest_us) else 0
    market = st.radio("Index", list(INDICES), index=default, format_func=INDICES.get, horizontal=True)
    with st.expander("Controls (re-run on the stored signals)", expanded=False):
        c1, c2, c3 = st.columns(3)
        kappa = c1.slider("Aggressiveness kappa", 0.0, 1.5, float(settings.get("module_a.kappa", 0.6)), 0.1)
        half = c2.select_slider("Half-life (hours)", options=[2.0, 6.0, 24.0, 72.0], value=6.0)
        strict = c3.toggle("Strict S&P DJI filters", value=False,
                           help="Relevance 100, novelty 100, |sentiment| >= 0.6, as in the S&P 500 RavenPack AI Sentiment Index")
        rerun = st.button("Apply controls")
    snaps = recompute(kappa, strict, half, market) if rerun else (india if market == "IN" else us)
    if not snaps:
        st.info("No index snapshots yet. Replay the DeepSeek or tariff pack (US) or the Adani pack (India) from the sidebar.")
        return
    tickers = list(snaps[0]["bench"])
    times = [pd.Timestamp(s["as_of"]) for s in snaps]
    last = snaps[-1]
    if last.get("inputs") == "fallback":
        st.caption("Inputs: equal benchmark weights and 25% volatility (run `make data` for market caps and EWMA volatility).")

    m = st.columns(5)
    m[0].metric("Rebalances", len(snaps) - 1)
    m[1].metric("Seismo turnover", f"{last['turnover_total']:.1%}")
    m[2].metric("Naive tilt turnover", f"{last['naive_turnover_total']:.1%}",
                help="Latest document sentiment, proportional reweighting, no filters, no caps")
    m[3].metric("Cost at 5 bp", f"{last['turnover_total'] * 5:.2f} bp", delta=f"naive {last['naive_turnover_total'] * 5:.2f} bp", delta_color="off")
    m[4].metric("Circuit breakers", ", ".join(last["breaker"]) or "none")

    moved = {t: max(abs(s["weights"][t] - s["bench"][t]) for s in snaps) for t in tickers}
    movers = [t for t in sorted(moved, key=lambda t: -moved[t]) if moved[t] > 0.002][:6]
    if movers:
        fig = go.Figure()
        for i, t in enumerate(sorted(movers, key=lambda t: min(s["weights"][t] - s["bench"][t] for s in snaps))):
            fig.add_trace(go.Scatter(x=times, y=[s["weights"][t] for s in snaps], name=short(t), mode="lines+markers",
                                     line={"shape": "hv", "width": 2.5, "color": PALETTE[i % len(PALETTE)]},
                                     hovertemplate=f"{t} %{{y:.2%}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[times[0], times[-1]], y=[last["bench"][t]] * 2, mode="lines", showlegend=False,
                                     line={"dash": "dot", "width": 1, "color": PALETTE[i % len(PALETTE)]}, hoverinfo="skip"))
        base_layout(fig, "Weights of the names that moved (dotted = benchmark)", 340)
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    fig = go.Figure()
    for i, t in enumerate(sorted(tickers, key=lambda t: -last["bench"][t])):
        fig.add_trace(go.Scatter(x=times, y=[s["weights"][t] for s in snaps], name=short(t), stackgroup="w",
                                 line={"shape": "hv", "width": 0.5, "color": PALETTE[i % len(PALETTE)]},
                                 hovertemplate=f"{t} %{{y:.2%}}<extra></extra>"))
    base_layout(fig, "Index weights over time (stacked; steps at each rebalance)", 360)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    left, right = st.columns(2, gap="medium")
    with left:
        active = {t: last["weights"][t] - last["bench"][t] for t in tickers}
        order = sorted(tickers, key=lambda t: active[t])
        fig = go.Figure(go.Bar(x=[active[t] for t in order], y=[short(t) for t in order], orientation="h",
                               marker_color=[NEG if active[t] < 0 else POS for t in order],
                               hovertemplate="%{y} %{x:+.2%}<extra></extra>"))
        base_layout(fig, "Active weight vs benchmark (latest)", 460)
        fig.update_xaxes(tickformat="+.1%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        z = pd.DataFrame([s["index"] for s in snaps], index=[t.strftime("%m-%d %H:%M") for t in times])[tickers]
        fig = go.Figure(go.Heatmap(z=z.T.values, x=z.index, y=[short(t) for t in tickers], zmin=-1, zmax=1,
                                   colorscale=[[0, NEG], [0.5, "#1B2638"], [1, POS]],
                                   colorbar={"title": "Sentiment", "tickfont": {"color": MUTED}}))
        base_layout(fig, "Entity sentiment index at each snapshot", 460)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    perf_section(snaps)

    st.markdown("**Why did the weights move?**")
    labels = [f"{t:%m-%d %H:%M} UTC - {s['reason']}" for t, s in zip(times, snaps, strict=True)]
    pick = st.selectbox("Rebalance", range(len(snaps)), index=len(snaps) - 1, format_func=lambda i: labels[i])
    snap = snaps[pick]
    attr = pd.DataFrame([{"Ticker": short(t), "_id": t, **{k: v for k, v in a.items() if k != "events"}, "Events": len(a["events"])}
                         for t, a in snap["attribution"].items()])
    attr["Change"] = attr["final"] - attr["prev"]
    attr = attr.reindex(attr["Change"].abs().sort_values(ascending=False).index)
    st.dataframe(
        attr[["Ticker", "bench", "prev", "signal_target", "final", "Change", "from_signal", "from_constraints", "breaker", "Events"]],
        hide_index=True, width="stretch", height=260,
        column_config={
            "bench": st.column_config.NumberColumn("Benchmark", format="percent"),
            "prev": st.column_config.NumberColumn("Before", format="percent"),
            "signal_target": st.column_config.NumberColumn("Signal-only target", format="percent"),
            "final": st.column_config.NumberColumn("After", format="percent"),
            "Change": st.column_config.NumberColumn(format="percent"),
            "from_signal": st.column_config.NumberColumn("From signal", format="percent", help="Signal-only target minus benchmark"),
            "from_constraints": st.column_config.NumberColumn("From constraints", format="percent", help="Name, active and sector caps pulled it back by this much"),
            "breaker": st.column_config.CheckboxColumn("Breaker"),
        },
    )
    movers = [t for t in attr["_id"] if snap["attribution"][t]["events"]]
    if movers:
        who = st.selectbox("Events behind", movers, format_func=short)
        st.caption(f"{who}: shrunk z-score {snap['z'].get(who, 0.0):+.2f}, Grinold alpha (IC x volatility x z) "
                   f"{snap['alpha'].get(who, 0.0):+.2%}")
        for e in snap["attribution"][who]["events"]:
            st.markdown(f"<blockquote class='evidence'>{e['span']}</blockquote>", unsafe_allow_html=True)
            st.caption(f"{e['publisher']} - {e['event'].lower()}, sentiment {e['sentiment']:+.2f}, impact {e['impact']}, "
                       f"{e['publishers']} independent publisher(s)")


def perf_section(snaps: list[dict]) -> None:
    from seismo.module_a.performance import replay_performance, summary

    paths = replay_performance(snaps, settings.root / "data" / "market")
    if paths is None:
        st.caption("Equity curves appear once data/market/prices_daily.csv covers the replay window (`make data`).")
        return
    fig = go.Figure()
    colours = {"Seismo": ACCENT, "Benchmark": MUTED, "Naive tilt": WATCH}
    for col in paths.columns:
        fig.add_trace(go.Scatter(x=paths.index, y=paths[col], name=col, line={"color": colours[col], "width": 2}))
    base_layout(fig, "Replay value (base 100): trades at the next open, 5 bp per unit turnover", 300)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    s = summary(paths)
    st.dataframe(pd.DataFrame(s).T.rename(columns={"return": "Return", "max_drawdown": "Max drawdown"}),
                 column_config={"Return": st.column_config.NumberColumn(format="percent"),
                                "Max drawdown": st.column_config.NumberColumn(format="percent")})
