from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["SEISMO_SENTIMENT_BACKEND"] = "lexicon"  # CI never downloads models
os.environ.setdefault("SEISMO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from seismo.config import load_settings  # noqa: E402
from seismo.engine import Engine  # noqa: E402
from seismo.nlp.sentiment import LexiconBackend  # noqa: E402
from seismo.schemas import Document, SourceType  # noqa: E402
from seismo.universe import Universe  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
DEMO_PACK = ROOT / "data" / "replay" / "demo_synthetic.jsonl"


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def universe(settings) -> Universe:
    return Universe.load(settings.path("universe.path"))


@pytest.fixture
def engine(universe, settings) -> Engine:
    return Engine(universe, LexiconBackend(), settings)


_counter = {"n": 0}


@pytest.fixture
def make_doc():
    def _make(
        title: str = "",
        body: str = "",
        source_type: SourceType = SourceType.NEWS,
        publisher: str = "test-wire.example",
        published_at: datetime | None = None,
        meta: dict | None = None,
    ) -> Document:
        _counter["n"] += 1
        return Document(
            doc_id=f"t_{_counter['n']}",
            source="test",
            source_type=source_type,
            publisher=publisher,
            published_at=published_at or datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            title=title,
            body=body,
            meta=meta or {},
        )

    return _make
