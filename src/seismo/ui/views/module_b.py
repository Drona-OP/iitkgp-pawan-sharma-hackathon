"""Module B page: trigger log, value waterfall, loss heatmap, ECL Sankey, CET1 gauge, reverse stress, memo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from seismo.ui.common import (
    ACCENT,
    DECISION_COLOUR,
    GLOSSARY,
    INK,
    MUTED,
    NEG,
    POS,
    WATCH,
    base_layout,
    header,
    records,
    settings,
)

FLOORS = {"RBI 8.0%": 0.08, "Basel 7.0%": 0.07}


@st.cache_resource
def get_engine():
    from seismo.module_b.stress import StressEngine

    return StressEngine.from_settings(settings)


def shocks_path() -> Path:
    p = Path(settings.get("module_b.shocks_path", "data/market/analog_shocks.csv"))
    return p if p.is_absolute() else settings.root / p


def bn(x: float) -> str:
    return f"${x / 1e9:,.2f}bn" if abs(x) >= 1e9 else f"${x / 1e6:,.0f}mn"


def waterfall(r: dict) -> go.Figure:
    before = sum(r["value_before"].values())
    after = sum(r["value_after"].values())
    order = ["Bonds", "Derivatives", "Equities", "Loans"]
    labels = ["Before"] + [f"{c} {'(ECL)' if c == 'Loans' else '(MTM)'}" for c in order] + ["After"]
    values = [before] + [r["pnl_by_class"].get(c, 0.0) for c in order] + [after]
    fig = go.Figure(go.Waterfall(
        x=labels, y=[v / 1e9 for v in values], measure=["absolute"] + ["relative"] * len(order) + ["total"],
        text=[bn(v) for v in values], textposition="outside",
        increasing={"marker": {"color": POS}}, decreasing={"marker": {"color": NEG}},
        totals={"marker": {"color": ACCENT}}, connector={"line": {"color": MUTED, "width": 1}},
    ))
    base_layout(fig, "Portfolio value before and after the stress (by asset class)", 340)
    lo = min(before, after) * 0.985 / 1e9
    fig.update_yaxes(range=[lo, max(before, after) * 1.006 / 1e9], tickprefix="$", ticksuffix="bn", tickformat=".1f")
    return fig


def heatmap(r: dict) -> go.Figure:
    df = pd.DataFrame(r["heatmap"])
    pivot = df.pivot_table(index="sector", columns="asset_class", values="loss", aggfunc="sum").fillna(0.0) / 1e6
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0, POS], [0.5, "#1B2638"], [1, NEG]], zmid=0,
        colorbar={"title": "$mn loss", "tickfont": {"color": MUTED}},
        hovertemplate="%{y} / %{x}: %{z:,.1f} $mn<extra></extra>",
    ))
    return base_layout(fig, "Loss by sector and asset class ($mn; red = loss)", 420)


def sankey(r: dict) -> go.Figure:
    flows = [f for f in r["stage_flows"] if f["ead"] > 0]
    left = ["Stage 1 (before)", "Stage 2 (before)", "Stage 3 (before)"]
    right = ["Stage 1 (after)", "Stage 2 (after)", "Stage 3 (after)"]
    idx = {name: i for i, name in enumerate(left + right)}
    colours = {"Stage 1": POS, "Stage 2": WATCH, "Stage 3": NEG}
    fig = go.Figure(go.Sankey(
        node={"label": left + right, "color": [colours[n[:7]] for n in left + right], "pad": 18,
              "line": {"color": "#101826", "width": 0.5}},
        link={"source": [idx[f"{f['from']} (before)"] for f in flows], "target": [idx[f"{f['to']} (after)"] for f in flows],
              "value": [f["ead"] / 1e6 for f in flows],
              "color": ["rgba(226,87,76,0.45)" if f["to"] > f["from"] else "rgba(79,163,165,0.35)" for f in flows],
              "customdata": [f["count"] for f in flows],
              "hovertemplate": "%{source.label} -> %{target.label}<br>%{value:,.0f} $mn EAD, %{customdata} facilities<extra></extra>"},
    ))
    return base_layout(fig, "ECL stage migration of the loan book (EAD, $mn)", 340)


def gauge(r: dict) -> go.Figure:
    after = r["cet1_ratio_after"] * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=after,
        number={"suffix": "%", "valueformat": ".2f", "font": {"color": INK}},
        delta={"reference": r["cet1_ratio_before"] * 100, "valueformat": ".2f", "suffix": " pp",
               "increasing": {"color": POS}, "decreasing": {"color": NEG}},
        gauge={"axis": {"range": [0, 16], "ticksuffix": "%", "tickcolor": MUTED},
               "bar": {"color": NEG if after < 8 else WATCH if after < 10 else POS},
               "bgcolor": "#172234",
               "steps": [{"range": [0, 7], "color": "rgba(226,87,76,0.25)"}, {"range": [7, 8], "color": "rgba(224,176,76,0.25)"}],
               "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.8, "value": r["cet1_ratio_before"] * 100}},
        title={"text": "CET1 ratio after stress (white line = before; shaded: below Basel 7% / RBI 8%)", "font": {"size": 13, "color": INK}},
    ))
    return base_layout(fig, None, 300)


def curve(r: dict) -> go.Figure:
    xs = [c[0] for c in r["curve"]]
    ys = [c[1] * 100 for c in r["curve"]]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line={"color": ACCENT, "width": 2}, name="CET1"))
    for name, v in FLOORS.items():
        fig.add_hline(y=v * 100, line_dash="dot", line_color=NEG if "RBI" in name else WATCH,
                      annotation_text=name, annotation_font_color=INK)
    if r.get("reverse_m_rbi"):
        fig.add_vline(x=r["reverse_m_rbi"], line_dash="dash", line_color=NEG,
                      annotation_text=f"breach at {r['reverse_m_rbi']:.2f}x", annotation_font_color=NEG)
    base_layout(fig, "Reverse stress: CET1 against severity (multiples of the analog)", 300)
    fig.update_xaxes(title="Severity multiple m")
    fig.update_yaxes(title="CET1 %", range=[min(0, min(ys) - 1), max(ys) + 1])
    return fig


def trigger_log() -> None:
    decisions = records("gate")
    st.markdown("**Trigger log** - every gate decision change; the naive column fires on any document with 10 x |sentiment| > 7")
    if not decisions:
        st.caption("No gate decisions yet. Replay the SVB, tariff or red-team pack.")
        return
    seismo = len({d["cluster_id"] for d in decisions if d["decision"] == "TRIGGER"})
    naive = len({d["cluster_id"] for d in decisions if d["naive_decision"] == "TRIGGER"})
    review = len({d["cluster_id"] for d in decisions if d["decision"] == "REVIEW"})
    retract = len({d["cluster_id"] for d in decisions if d["decision"] == "RETRACT"})
    c = st.columns(4)
    c[0].metric("Stories auto-triggered", seismo)
    c[1].metric("Stories the naive rule fires on", naive)
    c[2].metric("Stories sent to analyst review", review)
    c[3].metric("Stories retracted", retract)
    rows = []
    for d in decisions[::-1][:80]:
        failed = [x["name"] for x in d["checks"] if not x["passed"]]
        rows.append({"Time (UTC)": d["as_of"][:16].replace("T", " "), "Entity": d["entity_id"],
                     "Event": f"{d['event']['primary']}/{d['event'].get('subtype') or '-'}", "Impact": d["impact_score"],
                     "Seismo": d["decision"], "Naive": d["naive_decision"],
                     "Failed checks": ", ".join(failed) if d["decision"] != "TRIGGER" else "",
                     "Analog": d.get("scenario") or "", "Headline": (d.get("headline") or "")[:100]})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.map(lambda v: f"color: {DECISION_COLOUR.get(v, INK)}; font-weight: 600", subset=["Seismo", "Naive"]),
                 hide_index=True, width="stretch", height=280)


def render() -> None:
    header("Module B - Strategic stress test",
           "A corroborated high-impact event becomes a bank-grade stress result: the matching historical analog, "
           "scaled by impact, revalues a synthetic wholesale book; credit losses follow Vasicek PDs, rating migration "
           "and ECL staging; CET1 is checked against Basel and RBI floors; the reverse stress test finds the breaking point.")
    trigger_log()
    if not shocks_path().exists():
        st.warning("Module B needs data/market/analog_shocks.csv: run `make data` then `make shocks`.")
        return
    engine = get_engine()
    runs = records("stress")
    st.divider()
    st.markdown("**Stress result**")
    options = ["Custom scenario"] + [r["run_id"] for r in runs]
    labels = {r["run_id"]: f"{r['as_of'][:16].replace('T', ' ')} UTC - {r['scenario_label']} at impact {r['impact']}"
              f" ({r.get('epicenter') or 'market'})" for r in runs}
    default = len(options) - 1 if runs else 0
    pick = st.selectbox("Run", options, index=default, format_func=lambda o: labels.get(o, o))
    lib = engine.lib
    base = next((r for r in runs if r["run_id"] == pick), None)
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
    scen_ids = list(lib.scenarios)
    scenario = c1.selectbox("Analog", scen_ids, index=scen_ids.index(base["scenario"]) if base else 0,
                            format_func=lambda s: f"{lib.scenarios[s].label} ({lib.scenarios[s].start:%b %Y})")
    impact = c2.selectbox("Impact", [8, 9, 10], index=[8, 9, 10].index(base["impact"]) if base else 1)
    obligors = ["(none)"] + sorted(engine.book.obligors)
    epi_default = base["epicenter"] if base and base.get("epicenter") in engine.book.obligors else "(none)"
    epicenter = c3.selectbox("Obligor at the centre", obligors, index=obligors.index(epi_default),
                             format_func=lambda o: o if o == "(none)" else f"{o} - {engine.book.obligors[o].name}")
    m = c4.slider("Severity multiple m", 0.0, 3.0, 1.0, 0.05, help=GLOSSARY["Reverse stress"])
    event = (base.get("event") or "CREDIT_EVENT/BANK_RUN").split("/") if base else ["CREDIT_EVENT", "BANK_RUN"]
    from seismo.module_b.memo import write_memo
    from seismo.module_b.stress import StressRequest

    custom = (base is None or scenario != base["scenario"] or impact != base["impact"] or m != 1.0
              or (epicenter if epicenter != "(none)" else None) != base.get("epicenter"))
    if custom:
        req = StressRequest(scenario, impact, None if epicenter == "(none)" else epicenter, event[0], event[1] if len(event) > 1 else None, m)
        res = engine.run(req, "custom")
        r = res.model_dump(mode="json")
        r["memo"] = write_memo(res, base.get("trigger") if base else None)
    else:
        r = base

    before = sum(r["value_before"].values())
    after = sum(r["value_after"].values())
    k = st.columns(5)
    k[0].metric("Portfolio value", bn(after), delta=f"{(after - before) / before:+.2%}")
    k[1].metric("ECL (stress)", bn(r["ecl_after"]), delta=bn(r["ecl_after"] - r["ecl_before"]), delta_color="inverse", help=GLOSSARY["ECL"])
    k[2].metric("ECL (probability-weighted)", bn(r["ecl_weighted"]), help="Base 50% / adverse 30% / severe 20%, the RBI-style multi-scenario view")
    k[3].metric("CET1 ratio", f"{r['cet1_ratio_after']:.2%}", delta=f"{(r['cet1_ratio_after'] - r['cet1_ratio_before']) * 1e4:+.0f} bp", help=GLOSSARY["CET1"])
    k[4].metric("Breaches RBI 8% at", f"{r['reverse_m_rbi']:.2f}x" if r.get("reverse_m_rbi") else "> 10x", help=GLOSSARY["Reverse stress"])

    a, b = st.columns(2, gap="medium")
    a.plotly_chart(waterfall(r), width="stretch", config={"displayModeBar": False})
    b.plotly_chart(gauge(r), width="stretch", config={"displayModeBar": False})
    a, b = st.columns(2, gap="medium")
    a.plotly_chart(sankey(r), width="stretch", config={"displayModeBar": False})
    b.plotly_chart(curve(r), width="stretch", config={"displayModeBar": False})
    a, b = st.columns([3, 2], gap="medium")
    a.plotly_chart(heatmap(r), width="stretch", config={"displayModeBar": False})
    with b:
        st.markdown("**Top 10 contributors**")
        top = pd.DataFrame(r["top_contributors"])
        if not top.empty:
            top["loss"] = top["loss"] / 1e6
            st.dataframe(top[["name", "asset_class", "loss", "detail"]].rename(
                columns={"name": "Position", "asset_class": "Class", "loss": "Loss ($mn)", "detail": "Driver"}),
                hide_index=True, width="stretch", height=380,
                column_config={"Loss ($mn)": st.column_config.NumberColumn(format="%.1f")})
    st.markdown("**Risk memo** (template; every number injected from the run above)")
    st.code(r["memo"], language=None, wrap_lines=True)
    with st.expander("Shock vector used (analog x severity x m)"):
        from seismo.module_b.factors import shock_table

        src = {(row["scenario"], row["factor"]): row for row in shock_table(shocks_path())}
        rows = []
        for f, v in sorted(r["shocks"].items()):
            row = src.get((r["scenario"], f), {})
            unit = row.get("kind", "")
            rows.append({"Factor": f, "Shock": f"{v:+.2%}" if unit == "ret" else f"{v:+.0f} bp" if unit == "bp" else f"{v:+.1f} pts",
                         "Source series": row.get("source", ""), "Peak date": row.get("peak_date", "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Severity multiplier {r['multiplier']:.2f}; synthetic book; analog shocks computed from {shocks_path().name}.")
