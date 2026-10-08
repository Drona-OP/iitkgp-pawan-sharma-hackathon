"""Settings from config/seismo.yaml, with SEISMO_* environment overrides.

A dotted key maps to an env var: ``edgar.user_agent`` -> ``SEISMO_EDGAR_USER_AGENT``.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULTS: dict[str, Any] = {
    "store": {"path": "data/seismo.db"},
    "universe": {"path": "data/universe.csv"},
    "sentiment": {"backend": "auto", "finbert_model": "ProsusAI/finbert"},
    "linker": {"min_link_score": 0.6},
    "impact": {"model_path": "models/impact_calibrated.json"},
    "scenarios": {"path": "config/scenarios.yaml"},
    "module_a": {"kappa": 0.6, "strict": False},
    "module_b": {"shocks_path": "data/market/analog_shocks.csv"},
    "gate": {"min_impact": 8, "min_confidence": 0.7, "min_publishers": 3, "window_minutes": 60},
    "novelty": {"window_hours": 24.0},
    "aggregation": {"half_life_hours": 6.0, "impact_window_hours": 6.0, "history_hours": 72.0},
    "edgar": {
        "user_agent": "Seismo research contact@example.com",
        "poll_seconds": 60,
        "universe_only": True,
    },
    "gdelt": {
        "poll_seconds": 900,
        "timespan": "1h",
        "max_records": 250,
        "terms_per_query": 6,
        "pause_seconds": 6,
    },
    "bluesky": {
        "endpoint": "wss://jetstream2.us-east.bsky.network/subscribe",
        "max_cashtags": 5,
        "min_chars": 20,
        "max_posts_per_author_hour": 5,
    },
    "replay": {
        "default_pack": "data/replay/demo_synthetic.jsonl",
        "speed": 600.0,
        "max_gap_seconds": 2.0,
    },
    "api": {"host": "127.0.0.1", "port": 8000},
    "ui": {"port": 8501},
}


def find_root() -> Path:
    """Repo root: SEISMO_ROOT, else the nearest directory holding config/seismo.yaml, else CWD."""
    env = os.environ.get("SEISMO_ROOT")
    if env:
        return Path(env).resolve()
    for candidate in [Path.cwd(), *PACKAGE_DIR.parents]:
        if (candidate / "config" / "seismo.yaml").exists():
            return candidate
    return Path.cwd()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str, like: Any) -> Any:
    if isinstance(like, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(like, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


class Settings:
    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self.data = data
        self.root = root

    def _lookup(self, dotted: str, default: Any) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        env_key = "SEISMO_" + dotted.upper().replace(".", "_")
        value = self._lookup(dotted, default)
        if env_key in os.environ:
            return _coerce(os.environ[env_key], value)
        return value

    def path(self, dotted: str) -> Path:
        p = Path(str(self.get(dotted)))
        return p if p.is_absolute() else self.root / p


@lru_cache(maxsize=4)
def load_settings(config_path: str | None = None) -> Settings:
    root = find_root()
    path = Path(config_path or os.environ.get("SEISMO_CONFIG", root / "config" / "seismo.yaml"))
    data = DEFAULTS
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            data = _merge(DEFAULTS, yaml.safe_load(fh) or {})
    return Settings(data, root)
