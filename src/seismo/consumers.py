"""Builds the bus consumers for a run: the trigger gate (logged), Module B and Module A."""

from __future__ import annotations

import logging
from pathlib import Path

from seismo.scenarios import build_gate
from seismo.signals.gate import GateConsumer

log = logging.getLogger(__name__)


def build_consumers(settings, store, module_a: bool = True, module_b: bool = True) -> list:
    gate = GateConsumer(build_gate(settings))
    gate.sinks.append(lambda d: store.put_record("gate", d.decision_id, d.as_of, d.model_dump_json()))
    consumers: list = []
    if module_b:
        try:
            from seismo.module_b.consumer import ModuleBConsumer
            from seismo.module_b.stress import StressEngine

            shocks = Path(settings.get("module_b.shocks_path", "data/market/analog_shocks.csv"))
            if not shocks.is_absolute():
                shocks = settings.root / shocks
            if shocks.exists():
                consumers.append(ModuleBConsumer(StressEngine.from_settings(settings), store, gate))
            else:
                log.warning("Module B disabled: %s not found (run `make shocks`)", shocks)
        except Exception:  # noqa: BLE001 - the engine must run even if a module cannot start
            log.exception("Module B failed to start")
    consumers.append(gate)
    if module_a:
        try:
            from seismo.module_a.consumer import ModuleAConsumer

            consumers.append(ModuleAConsumer.from_settings(settings, store))
        except Exception:  # noqa: BLE001
            log.exception("Module A failed to start")
    return consumers
