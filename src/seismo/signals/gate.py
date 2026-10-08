"""The trigger gate: an automatic stress test needs every check to pass; anything else goes to review.

1. Impact >= 8 (the brief's "impact > 7").
2. The event class and subtype map to a historical analog in the scenario library.
3. Event-class confidence >= 0.7.
4. Corroboration: >= 3 independent publishers within 60 minutes, or one authoritative source
   (SEC filing, regulator, central bank, issuer statement, major wire). Coordinated reposts and
   lookalike domains never count.
5. Not disputed: no official denial matched to the story.

The naive comparator fires on any single document whose 10 x |sentiment| exceeds 7, which is
what an LLM-score-and-threshold pipeline does.
"""

from __future__ import annotations

from dataclasses import dataclass

from seismo.schemas import EventClass, GateCheck, GateDecision, Signal, stable_id


@dataclass(frozen=True)
class GateConfig:
    min_impact: int = 8
    min_confidence: float = 0.7
    min_publishers: int = 3
    window_minutes: int = 60
    naive_threshold: int = 7


class ScenarioMap:
    """(class, subtype) -> scenario id, with a class-level fallback key "CLASS/*"."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = dict(mapping)

    def lookup(self, primary: EventClass, subtype: str | None) -> str | None:
        if subtype and f"{primary.value}/{subtype}" in self.mapping:
            return self.mapping[f"{primary.value}/{subtype}"]
        return self.mapping.get(f"{primary.value}/*")


def naive_impact(sentiment: float) -> int:
    return max(1, min(10, round(10 * abs(sentiment))))


class TriggerGate:
    def __init__(self, scenarios: ScenarioMap, config: GateConfig | None = None) -> None:
        self.scenarios = scenarios
        self.cfg = config or GateConfig()

    def evaluate(self, sig: Signal, trigger_doc: Signal | None = None) -> GateDecision:
        cfg = self.cfg
        c = sig.corroboration
        scenario = self.scenarios.lookup(sig.event.primary, sig.event.subtype)
        corroborated = c.publishers_60m >= cfg.min_publishers or c.authoritative
        notes = []
        if c.coordinated:
            notes.append("coordinated reposts collapsed to one voice")
        if "lookalike_source" in sig.flags:
            notes.append("lookalike domain ignored")
        corr_detail = (
            f"{c.publishers_60m} independent publisher(s) in {cfg.window_minutes} min "
            f"(need {cfg.min_publishers}); authoritative source: {'yes' if c.authoritative else 'no'}"
            + (f"; {'; '.join(notes)}" if notes else "")
        )
        checks = [
            GateCheck(name="impact", passed=sig.impact_score >= cfg.min_impact,
                      detail=f"impact {sig.impact_score} (need >= {cfg.min_impact})"),
            GateCheck(name="scenario", passed=scenario is not None,
                      detail=f"{sig.event.primary.value}/{sig.event.subtype or '-'} -> {scenario or 'no analog'}"),
            GateCheck(name="confidence", passed=sig.event.confidence >= cfg.min_confidence,
                      detail=f"event confidence {sig.event.confidence:.2f} (need >= {cfg.min_confidence})"),
            GateCheck(name="corroboration", passed=corroborated, detail=corr_detail),
            GateCheck(name="not_disputed", passed=not c.disputed,
                      detail="official denial matched" if c.disputed else "no denial"),
        ]
        if sig.status == "retracted":
            decision = "RETRACT"
        elif sig.impact_score < cfg.min_impact:
            decision = "LOG"
        elif all(ch.passed for ch in checks):
            decision = "TRIGGER"
        else:
            decision = "REVIEW"

        doc_sent = trigger_doc.sentiment_score if trigger_doc is not None else sig.sentiment_score
        naive = "TRIGGER" if naive_impact(doc_sent) > cfg.naive_threshold and sig.status != "retracted" else "LOG"
        headline = sig.evidence[0].title or sig.evidence[0].span if sig.evidence else None
        return GateDecision(
            decision_id=stable_id("gate", sig.signal_id),
            as_of=sig.as_of,
            signal_id=sig.signal_id,
            cluster_id=sig.cluster_id,
            entity_id=sig.entity.id,
            event=sig.event,
            impact_score=sig.impact_score,
            decision=decision,  # type: ignore[arg-type]
            checks=checks,
            naive_decision=naive,  # type: ignore[arg-type]
            scenario=scenario,
            headline=headline,
            synthetic=sig.synthetic,
        )


class GateConsumer:
    """Evaluates every event-grain signal, logs decision changes, and fans decisions out."""

    def __init__(self, gate: TriggerGate, sinks: list | None = None) -> None:
        self.gate = gate
        self.sinks = list(sinks or [])   # callables taking a GateDecision
        self._last_doc: dict[str, Signal] = {}
        self._last_state: dict[str, tuple[str, str, int]] = {}
        self.log: list[GateDecision] = []

    def on_signal(self, sig: Signal) -> GateDecision | None:
        if sig.grain == "document" and sig.cluster_id:
            self._last_doc[sig.cluster_id] = sig
            return None
        if sig.grain != "event" or not sig.cluster_id:
            return None
        decision = self.gate.evaluate(sig, self._last_doc.get(sig.cluster_id))
        state = (decision.decision, decision.naive_decision, decision.impact_score)
        if self._last_state.get(sig.cluster_id) == state:
            return None
        self._last_state[sig.cluster_id] = state
        self.log.append(decision)
        for sink in self.sinks:
            sink(decision)
        return decision
