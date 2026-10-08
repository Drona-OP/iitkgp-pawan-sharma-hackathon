from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from seismo.schemas import (
    SCHEMA_VERSION,
    Corroboration,
    EntityRef,
    EventClass,
    EventLabel,
    Signal,
    SourceType,
    stable_id,
)


def _signal(**overrides):
    base = dict(
        signal_id="sig_x", grain="document", as_of=datetime(2026, 1, 1, tzinfo=UTC),
        entity=EntityRef(type="company", id="NVDA", name="NVIDIA Corp."),
        sentiment_score=-0.5, sentiment_confidence=0.7,
        event=EventLabel(primary=EventClass.CREDIT_EVENT, confidence=0.9),
        impact_score=8, novelty=100, relevance=90,
        corroboration=Corroboration(independent_publishers=2, source_types=[SourceType.NEWS]),
    )
    base.update(overrides)
    return Signal(**base)


def test_valid_signal_round_trips_json():
    sig = _signal()
    assert Signal.model_validate_json(sig.model_dump_json()) == sig
    assert sig.schema_version == SCHEMA_VERSION == "1.1"


@pytest.mark.parametrize("field,value", [("sentiment_score", 1.5), ("impact_score", 0), ("impact_score", 11), ("novelty", 101)])
def test_out_of_range_fields_are_rejected(field, value):
    with pytest.raises(ValidationError):
        _signal(**{field: value})


def test_naive_timestamps_become_utc():
    sig = _signal(as_of=datetime(2026, 1, 1, 9, 30))
    assert sig.as_of.tzinfo is not None


def test_stable_id_is_deterministic():
    assert stable_id("d", "a", 1) == stable_id("d", "a", 1)
    assert stable_id("d", "a", 1) != stable_id("d", "a", 2)


def test_json_schema_exposes_the_required_fields():
    props = Signal.model_json_schema()["properties"]
    for name in ("sentiment_score", "event", "impact_score", "evidence", "corroboration"):
        assert name in props
