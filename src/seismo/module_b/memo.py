"""The risk memo: a fixed template, every number injected from the stress result, every claim linked
to its evidence. No language model writes a number here; an optional LLM pass may only rephrase.
"""

from __future__ import annotations

from seismo.module_b.stress import RBI_FLOOR, StressResult


def _bn(x: float) -> str:
    return f"${x / 1e9:,.2f}bn" if abs(x) >= 1e9 else f"${x / 1e6:,.0f}mn"


def write_memo(r: StressResult, trigger: dict | None = None) -> str:
    trigger = trigger or {}
    before = sum(r.value_before.values())
    after = sum(r.value_after.values())
    worst_cls = min(r.pnl_by_class.items(), key=lambda kv: kv[1]) if r.pnl_by_class else ("-", 0.0)
    flows = {(f["from"], f["to"]): f for f in r.stage_flows}
    to2 = sum(f["count"] for (a, b), f in flows.items() if b == "Stage 2" and a == "Stage 1")
    to3 = sum(f["count"] for (a, b), f in flows.items() if b == "Stage 3" and a != "Stage 3")
    ead2 = sum(f["ead"] for (a, b), f in flows.items() if b == "Stage 2" and a == "Stage 1")
    breach = "remains above" if r.cet1_ratio_after > RBI_FLOOR else "falls below"
    lines = [
        f"STRESS MEMO - {r.scenario_label} analog at impact {r.impact} (severity x{r.multiplier:.2f})",
        "",
        "Trigger: " + (
            f"\"{trigger.get('headline', '')}\" - {trigger.get('entity', r.epicenter or 'market')}, "
            f"{trigger.get('event', r.event or '')}; {trigger.get('publishers', 0)} independent publishers, "
            f"authoritative source: {'yes' if trigger.get('authoritative') else 'no'}; gate passed "
            f"{trigger.get('checks_passed', 5)}/5." if trigger else "manual run from the dashboard."
        ),
        "",
        f"1. Portfolio value moves from {_bn(before)} to {_bn(after)} ({(after - before) / before:+.2%}). "
        f"The largest hit is {worst_cls[0].lower()} ({_bn(worst_cls[1])}).",
        f"2. Expected credit loss rises from {_bn(r.ecl_before)} to {_bn(r.ecl_after)} under the stress scenario "
        f"({_bn(r.ecl_weighted)} probability-weighted across base, adverse and severe). "
        f"{to2} facilities ({_bn(ead2)} EAD) move to Stage 2 and {to3} to Stage 3"
        + (f"; {', '.join(r.defaulted)} assumed in default." if r.defaulted else "."),
        f"3. CET1 falls from {r.cet1_ratio_before:.2%} to {r.cet1_ratio_after:.2%} "
        f"({(r.cet1_ratio_after - r.cet1_ratio_before) * 1e4:+.0f} bp) and {breach} the RBI 8.0% floor "
        f"(Basel minimum plus buffer 7.0%). RWA moves from {_bn(r.rwa_before)} to {_bn(r.rwa_after)} as ratings migrate.",
        "4. Reverse stress: " + (
            f"this book breaches 8.0% CET1 at {r.reverse_m_rbi:.2f}x the {r.scenario_label} analog"
            + (f" and 7.0% at {r.reverse_m_basel:.2f}x." if r.reverse_m_basel else ".")
            if r.reverse_m_rbi else "the book does not breach 8.0% CET1 even at 10x this analog."
        ),
        "5. Largest contributors: " + "; ".join(f"{c.name} ({c.asset_class}, {_bn(c.loss)})" for c in r.top_contributors[:5]) + ".",
        "",
        "Evidence: " + (" | ".join(f"{e.get('publisher')}: \"{e.get('span')}\"" for e in trigger.get("evidence", [])[:3]) or "n/a"),
        "",
        "Assumptions: synthetic book and internal ratings; analog shocks computed from public market data; "
        "staging thresholds are illustrative proxies, not RBI's exact rules; 25% tax on losses.",
    ]
    return "\n".join(lines)
