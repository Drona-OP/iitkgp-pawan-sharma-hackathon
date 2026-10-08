"""Scenario library loader (config/scenarios.yaml), shared by the trigger gate and Module B."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from seismo.signals.gate import GateConfig, ScenarioMap, TriggerGate


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    start: date
    end: date
    captures: str
    epicenters: tuple[str, ...] = ()   # names whose own observed move already holds the idiosyncratic shock
    scope: str = "global"              # "india": only Indian factors move (a local shock; US noise is dropped)


@dataclass(frozen=True)
class ScenarioLibrary:
    scenarios: dict[str, Scenario]
    event_map: dict[str, str]
    severity_multiplier: dict[int, float]
    vasicek_z: dict[int, float]
    idiosyncratic: dict[str, dict[str, float]]
    ecl_weights: dict[str, float]
    default_subtypes: tuple[str, ...] = ()
    contagion_share: float = 0.34
    group_share: float = 0.67

    def multiplier(self, impact: int) -> float:
        keys = sorted(self.severity_multiplier)
        impact = max(keys[0], min(keys[-1], impact))
        return self.severity_multiplier[impact]

    def z(self, impact: int) -> float:
        keys = sorted(self.vasicek_z)
        impact = max(keys[0], min(keys[-1], impact))
        return self.vasicek_z[impact]

    def idio(self, event_class: str) -> dict[str, float]:
        return self.idiosyncratic.get(event_class, self.idiosyncratic.get("default", {}))

    def scenario_map(self) -> ScenarioMap:
        return ScenarioMap(self.event_map)


def _d(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


@lru_cache(maxsize=4)
def load_library(path: str | Path) -> ScenarioLibrary:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenarios = {
        sid: Scenario(sid, s["label"], _d(s["start"]), _d(s["end"]), s.get("captures", ""),
                      tuple(s.get("epicenters", []) or []), str(s.get("scope", "global")))
        for sid, s in raw["scenarios"].items()
    }
    sev = raw.get("severity", {})
    return ScenarioLibrary(
        scenarios=scenarios,
        event_map={str(k): str(v) for k, v in raw.get("event_map", {}).items()},
        severity_multiplier={int(k): float(v) for k, v in sev.get("multiplier", {8: 0.75, 9: 1.0, 10: 1.25}).items()},
        vasicek_z={int(k): float(v) for k, v in sev.get("vasicek_z", {8: -1.5, 9: -2.0, 10: -2.33}).items()},
        idiosyncratic={k: dict(v) for k, v in raw.get("idiosyncratic", {}).items()},
        ecl_weights={k: float(v) for k, v in raw.get("ecl_weights", {"base": 0.5, "adverse": 0.3, "severe": 0.2}).items()},
        default_subtypes=tuple(raw.get("default_subtypes", [])),
        contagion_share=float(raw.get("contagion_share", 0.34)),
        group_share=float(raw.get("group_share", 0.67)),
    )


def build_gate(settings) -> TriggerGate:
    lib = load_library(str(settings.path("scenarios.path")))
    cfg = GateConfig(
        min_impact=int(settings.get("gate.min_impact", 8)),
        min_confidence=float(settings.get("gate.min_confidence", 0.7)),
        min_publishers=int(settings.get("gate.min_publishers", 3)),
        window_minutes=int(settings.get("gate.window_minutes", 60)),
    )
    return TriggerGate(lib.scenario_map(), cfg)
