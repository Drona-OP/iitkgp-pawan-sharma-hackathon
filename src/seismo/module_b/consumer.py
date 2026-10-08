"""Module B as a bus consumer: gate decisions in, stress runs out.

It listens to the trigger gate. A TRIGGER for a story that has not triggered before (or whose
impact rose since) runs the matching analog; REVIEW decisions are queued for an analyst;
RETRACT decisions mark earlier runs on that story as withdrawn.
"""

from __future__ import annotations

import json
import logging

from seismo.module_b.memo import write_memo
from seismo.module_b.stress import StressEngine, StressRequest
from seismo.schemas import GateDecision, Signal, stable_id

log = logging.getLogger(__name__)


class ModuleBConsumer:
    """Register before the gate consumer so the event signal is known when its decision arrives."""

    def __init__(self, engine: StressEngine, store, gate_consumer) -> None:
        self.engine = engine
        self.store = store
        self._fired: dict[str, int] = {}
        self._signals: dict[str, Signal] = {}
        gate_consumer.sinks.append(self.on_decision)

    def on_signal(self, sig: Signal) -> None:
        if sig.grain == "event" and sig.cluster_id:
            self._signals[sig.signal_id] = sig

    def on_decision(self, d: GateDecision) -> None:
        if d.decision == "RETRACT" and d.cluster_id in self._fired:
            self.store.put_record("stress_retraction", d.decision_id, d.as_of, d.model_dump_json())
            return
        if d.decision != "TRIGGER" or not d.scenario:
            return
        if self._fired.get(d.cluster_id or "", 0) >= d.impact_score:
            return
        self._fired[d.cluster_id or ""] = d.impact_score
        sig = self._signals.get(d.signal_id)
        epicenter = d.entity_id if d.entity_id in self.engine.book.obligors else None
        req = StressRequest(d.scenario, d.impact_score, epicenter, d.event.primary.value, d.event.subtype)
        run_id = stable_id("stress", d.decision_id)
        result = self.engine.run(req, run_id)
        trigger = {
            "headline": d.headline, "entity": d.entity_id,
            "event": f"{d.event.primary.value}/{d.event.subtype}",
            "publishers": sig.corroboration.independent_publishers if sig else 0,
            "authoritative": sig.corroboration.authoritative if sig else False,
            "checks_passed": sum(c.passed for c in d.checks),
            "evidence": [e.model_dump(mode="json") for e in (sig.evidence if sig else [])],
            "decision_id": d.decision_id, "signal_id": d.signal_id, "cluster_id": d.cluster_id,
        }
        result.as_of = d.as_of
        result.trigger = trigger
        result.synthetic = True
        result.memo = write_memo(result, trigger)
        self.store.put_record("stress", run_id, d.as_of, result.model_dump_json())
        log.info("Stress run %s: %s impact %d -> CET1 %.2f%%", run_id, d.scenario, d.impact_score,
                 100 * result.cet1_ratio_after)


def load_runs(store) -> list[dict]:
    return [json.loads(r) for r in store.records("stress")]
