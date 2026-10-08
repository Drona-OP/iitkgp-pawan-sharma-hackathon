"""`make results`: regenerate every number in the deck and README from the repository.

    python -m seismo.eval.results     # writes docs/results/results.json and results.md

- Replay scorecard: each pack through the full pipeline, gated trigger vs the naive trigger
  ("fire on any document with 10 x |sentiment| > 7"), with timing against a reference moment.
- Dedup and corroboration: reports counted naively vs independent voices.
- Gold set: entity linking vs exact string match; entity-window sentiment vs whole-document
  sentiment; rule-based event class vs the majority class.
- Module A: underweight timing and turnover vs the naive tilt; replay returns if prices exist.
- Module B: the first stress run per pack (CET1 before and after, reverse-stress multiple).
- Latency: engine and pipeline time per document on this CPU.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from seismo.config import load_settings
from seismo.engine import Engine
from seismo.ingest.replay import load_pack
from seismo.module_a.consumer import ModuleAConsumer
from seismo.nlp.events import classify_event
from seismo.nlp.linker import EntityLinker
from seismo.nlp.sentiment import make_backend
from seismo.scenarios import build_gate
from seismo.schemas import Document, SourceType
from seismo.signals.gate import GateConsumer, naive_impact
from seismo.universe import Universe

PACKS = {
    # pack: (should auto-trigger?, reference moment, what the reference is)
    "svb_2023": (True, "2023-03-10T16:15:00Z", "regulators close SVB"),
    "deepseek_2025": (True, "2025-01-27T14:30:00Z", "Monday open"),
    "tariff_2025": (True, "2025-04-03T13:30:00Z", "first open after the announcement"),
    "red_team": (False, "2026-09-15T14:35:00Z", "official denial"),
    "quiet_day": (False, None, ""),
}


class MemoryRecords:
    def __init__(self) -> None:
        self.rows: dict[str, list[str]] = {}

    def put_record(self, kind: str, key: str, as_of, payload: str) -> None:
        self.rows.setdefault(kind, []).append(payload)


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _hours(a: datetime | None, b: datetime | None) -> str:
    """How long before (or after) the reference moment b the event a happened."""
    if a is None or b is None:
        return "-"
    h = (b - a).total_seconds() / 3600
    amount = f"{abs(h):.1f} h" if abs(h) < 72 else f"{abs(h) / 24:.1f} d"
    return f"{amount} {'before' if h >= 0 else 'after'}"


def run_pack(name: str, settings, universe: Universe, backend, stress_engine) -> dict:
    engine = Engine(universe, backend, settings)
    gate = GateConsumer(build_gate(settings))
    store = MemoryRecords()
    module_a = ModuleAConsumer.from_settings(settings, store=None)
    gate.sinks.append(module_a.on_decision)
    module_b = None
    if stress_engine is not None:
        from seismo.module_b.consumer import ModuleBConsumer

        module_b = ModuleBConsumer(stress_engine, store, gate)
    docs = load_pack(settings.root / "data" / "replay" / f"{name}.jsonl")
    engine_ms, naive_fires, first_naive = [], 0, None
    news_reports = 0
    stories = {}
    for doc in docs:
        t0 = time.perf_counter()
        result = engine.process_full(doc)
        engine_ms.append((time.perf_counter() - t0) * 1000)
        if doc.source_type != SourceType.SOCIAL:
            news_reports += 1
        fired = any(naive_impact(s.sentiment_score) > 7 for s in result.doc_signals)
        if fired:
            naive_fires += 1
            first_naive = first_naive or doc.published_at
        for sig in result.all:
            if module_b is not None:
                module_b.on_signal(sig)
            gate.on_signal(sig)
            module_a.on_signal(sig)
            if sig.grain == "event":
                stories[sig.cluster_id] = sig
    log = gate.log
    triggers = [d for d in log if d.decision == "TRIGGER"]
    expected, ref, ref_label = PACKS[name]
    ref_t = _ts(ref)
    first_trigger = triggers[0].as_of if triggers else None
    severe = [s for s in stories.values() if s.impact_score >= 8]
    runs = [json.loads(r) for r in store.rows.get("stress", [])]
    out = {
        "pack": name,
        "documents": len(docs),
        "expected_trigger": expected,
        "naive_fires": naive_fires,
        "seismo_triggers": len({d.cluster_id for d in triggers}),
        "reviews": len({d.cluster_id for d in log if d.decision == "REVIEW"}),
        "retractions": sum(d.decision == "RETRACT" for d in log),
        "first_trigger": first_trigger.isoformat() if first_trigger else None,
        "first_naive": first_naive.isoformat() if first_naive else None,
        "reference": ref_label,
        "trigger_vs_reference": _hours(first_trigger, ref_t) if expected else "-",
        "correct": (len(triggers) > 0) == expected,
        "stories": len(stories),
        "severe_stories": len(severe),
        "news_reports": news_reports,
        "social_posts": len(docs) - news_reports,
        "independent_publishers_max": max((s.corroboration.independent_publishers for s in stories.values()), default=0),
        "coordinated_groups": sum(s.corroboration.coordinated for s in stories.values()),
        "engine_ms": engine_ms,
        "module_a": {
            "rebalances": max(0, len(module_a.snapshots) - 1),
            "turnover": round(module_a.turnover_total, 4),
            "naive_turnover": round(module_a.naive_turnover, 4),
            "snapshots": module_a.snapshots,
        },
        "module_b": [{k: r[k] for k in ("scenario", "impact", "epicenter", "cet1_ratio_before", "cet1_ratio_after",
                                         "ecl_before", "ecl_after", "reverse_m_rbi", "pre_tax_loss")}
                     for r in sorted(runs, key=lambda r: r["cet1_ratio_after"])[:3]],
    }
    if name == "deepseek_2025":
        before_open = [s for s in module_a.snapshots if s["as_of"] < "2025-01-27T14:30"]
        if before_open and "NVDA" in before_open[-1]["weights"]:
            s = before_open[-1]
            first_cut = next((x for x in module_a.snapshots if x["weights"]["NVDA"] < x["bench"]["NVDA"] - 1e-6), None)
            out["module_a"]["nvda_before_open"] = {"weight": s["weights"]["NVDA"], "bench": s["bench"]["NVDA"],
                                                   "first_cut": first_cut["as_of"] if first_cut else None,
                                                   "lead_vs_open": _hours(_ts(first_cut["as_of"]) if first_cut else None, ref_t)}
    return out


def gold_metrics(settings, universe: Universe, backend) -> dict:
    path = settings.root / "data" / "gold" / "headlines.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    linker = EntityLinker(universe, float(settings.get("linker.min_link_score", 0.6)))
    engine = Engine(universe, backend, settings)
    tp = fp = fn = btp = bfp = bfn = 0
    win_ok = doc_ok = pairs = 0
    ev_ok = 0
    labels = Counter(r["event"] for r in rows)
    majority = labels.most_common(1)[0][0]
    for i, r in enumerate(rows):
        doc = Document(doc_id=f"g{i}", source="gold", source_type=SourceType.NEWS, publisher="gold.example",
                       published_at=datetime(2026, 1, 1, tzinfo=UTC), title=r["text"])
        gold = {k for k in r["entities"] if not k.startswith("MACRO:")}
        pred = {m.entity_id for m in linker.link(doc) if m.entity_type == "company"}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        # Baseline: exact, case-insensitive match of any alias, including ambiguous ones, no context.
        low = r["text"].lower()
        base = {e.entity_id for e in universe.companies()
                if any(a.lower() in low for a in (*e.aliases, *e.ambiguous_aliases))}
        btp += len(gold & base)
        bfp += len(base - gold)
        bfn += len(gold - base)
        sigs = {s.entity.id: s.sentiment_score for s in engine.process(doc)}
        whole = backend.predict([r["text"]])[0]
        whole_s = whole[2] - whole[0]
        for ent, lab in r["entities"].items():
            if ent not in sigs:
                continue
            pairs += 1
            win_ok += int(_sign(sigs[ent]) == lab)
            doc_ok += int(_sign(whole_s) == lab)
        ev_ok += int(classify_event(doc).primary.value == r["event"])
    p = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    bp = btp / max(1, btp + bfp)
    br = btp / max(1, btp + bfn)
    n = len(rows)
    return {
        "table": [
            {"Component": "Entity linking", "Metric": "precision / recall", "Naive baseline": f"{bp:.2f} / {br:.2f} (exact alias match)",
             "Seismo": f"{p:.2f} / {rc:.2f} (alias + context + cashtags)"},
            {"Component": "Target sentiment", "Metric": f"sign accuracy on {pairs} entity pairs",
             "Naive baseline": f"{doc_ok / max(1, pairs):.2f} (whole-document score)", "Seismo": f"{win_ok / max(1, pairs):.2f} (entity-window score)"},
            {"Component": "Event class", "Metric": f"accuracy on {n} headlines",
             "Naive baseline": f"{labels[majority] / n:.2f} (majority class)", "Seismo": f"{ev_ok / n:.2f} (8-K items + weighted rules)"},
        ],
        "note": f"{n} author-labelled illustrative headlines in data/gold/headlines.jsonl, incl. ambiguous names and "
                "two-company headlines with opposite sentiment. Small and in-sample for the rules; read as a sanity check.",
    }


def _sign(x: float, eps: float = 0.1) -> int:
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def main() -> int:
    settings = load_settings()
    universe = Universe.load(settings.path("universe.path"))
    backend = make_backend(str(settings.get("sentiment.backend", "auto")), str(settings.get("sentiment.finbert_model")))
    stress_engine = None
    shocks = Path(settings.get("module_b.shocks_path", "data/market/analog_shocks.csv"))
    shocks = shocks if shocks.is_absolute() else settings.root / shocks
    if shocks.exists():
        from seismo.module_b.stress import StressEngine

        stress_engine = StressEngine.from_settings(settings)
    packs = [run_pack(name, settings, universe, backend, stress_engine) for name in PACKS]
    all_ms = [m for p in packs for m in p["engine_ms"]]
    probe = Engine(universe, backend, settings)
    impact_name = probe.impact.name
    latency = {
        "documents": len(all_ms),
        "engine_p50_ms": round(statistics.median(all_ms), 2),
        "engine_p95_ms": round(statistics.quantiles(all_ms, n=20)[-1], 2),
        "docs_per_second": round(1000 * len(all_ms) / sum(all_ms), 1),
        "llm_calls": 0,
    }
    scorecard = [{
        "Pack": p["pack"].replace("_", " "),
        "Should trigger": "yes" if p["expected_trigger"] else "no",
        "Naive stress tests": p["naive_fires"],
        "Seismo auto-triggers": p["seismo_triggers"],
        "Held for review": p["reviews"],
        "Retractions": p["retractions"],
        "First trigger vs reference": f"{p['trigger_vs_reference']} ({p['reference']})" if p["expected_trigger"] else "-",
        "Correct": "yes" if p["correct"] else "NO",
    } for p in packs]
    dedup = [{
        "Pack": p["pack"].replace("_", " "),
        "Documents": p["documents"],
        "News/filing reports": p["news_reports"],
        "Social posts": p["social_posts"],
        "Stories (events)": p["stories"],
        "Max independent publishers": p["independent_publishers_max"],
        "Coordinated groups": p["coordinated_groups"],
    } for p in packs]
    perf = None
    ds = next(p for p in packs if p["pack"] == "deepseek_2025")
    try:
        from seismo.module_a.performance import replay_performance, summary

        paths = replay_performance(ds["module_a"]["snapshots"], settings.root / "data" / "market")
        perf = summary(paths) if paths is not None else None
    except Exception:  # noqa: BLE001
        perf = None
    out = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "sentiment_backend": backend.name,
        "impact_model": impact_name,
        "scorecard": scorecard,
        "dedup": dedup,
        "gold": gold_metrics(settings, universe, backend),
        "latency": latency,
        "module_a": {p["pack"]: {k: v for k, v in p["module_a"].items() if k != "snapshots"} for p in packs},
        "module_a_deepseek_performance": perf,
        "module_b": {p["pack"]: p["module_b"] for p in packs},
        "stress_inputs": "data/market/analog_shocks.csv" if stress_engine else None,
    }
    results = settings.root / "docs" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "results.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (results / "results.md").write_text(to_markdown(out), encoding="utf-8")
    print(to_markdown(out))
    return 0


def _table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def to_markdown(out: dict) -> str:
    parts = [
        f"# Seismo results (generated {out['generated_at']})",
        f"Sentiment backend `{out['sentiment_backend']}`, impact model `{out['impact_model']}`. Regenerate with `make results`.",
        "## Replay scorecard: gated vs naive trigger", _table(out["scorecard"]),
        "## Dedup and corroboration", _table(out["dedup"]),
        "## Gold set", _table(out["gold"]["table"]), out["gold"]["note"],
        "## Latency (CPU)", _table([out["latency"]]),
    ]
    ma = out["module_a"].get("deepseek_2025", {})
    if ma.get("nvda_before_open"):
        n = ma["nvda_before_open"]
        parts += ["## Module A (DeepSeek replay)",
                  f"NVDA first cut at {n['first_cut']} ({n['lead_vs_open']} the Monday open); weight before the open "
                  f"{n['weight']:.2%} vs benchmark {n['bench']:.2%}. Turnover {ma['turnover']:.1%} vs naive tilt {ma['naive_turnover']:.1%}."]
    if out.get("module_a_deepseek_performance"):
        parts += [_table([{"Portfolio": k, "Return": f"{v['return']:+.2%}", "Max drawdown": f"{v['max_drawdown']:.2%}"}
                          for k, v in out["module_a_deepseek_performance"].items()])]
    mb = [{"Pack": k, **r} for k, rs in out["module_b"].items() for r in rs[:1]]
    if mb:
        parts += ["## Module B (most severe auto-triggered stress run per pack)", _table([{
            "Pack": r["Pack"], "Analog": r["scenario"], "Impact": r["impact"], "Obligor": r["epicenter"] or "-",
            "CET1 before": f"{r['cet1_ratio_before']:.2%}", "CET1 after": f"{r['cet1_ratio_after']:.2%}",
            "ECL": f"${r['ecl_before'] / 1e6:,.0f}mn -> ${r['ecl_after'] / 1e6:,.0f}mn",
            "Breaches 8% at": f"{r['reverse_m_rbi']:.2f}x" if r["reverse_m_rbi"] else "> 10x"} for r in mb])]
    return "\n\n".join(p for p in parts if p) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
