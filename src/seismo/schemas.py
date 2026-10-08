"""Typed contracts for everything that crosses a module boundary (signal schema v1)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic short id, so a replay produces identical ids run after run."""
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


class SourceType(StrEnum):
    NEWS = "news"
    SOCIAL = "social"
    FILING = "filing"


class EventClass(StrEnum):
    MACROECONOMIC = "MACROECONOMIC"
    GEOPOLITICAL = "GEOPOLITICAL"
    CREDIT_EVENT = "CREDIT_EVENT"
    MA_CORPORATE_ACTION = "MA_CORPORATE_ACTION"
    PRODUCT_STRATEGY = "PRODUCT_STRATEGY"
    EARNINGS_GUIDANCE = "EARNINGS_GUIDANCE"
    LEGAL_REGULATORY = "LEGAL_REGULATORY"
    OPERATIONAL_ESG = "OPERATIONAL_ESG"
    MANAGEMENT_GOVERNANCE = "MANAGEMENT_GOVERNANCE"
    OTHER = "OTHER"


class Document(BaseModel):
    """One normalized text item from any source."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    source: str
    source_type: SourceType
    publisher: str
    author: str | None = None
    url: str | None = None
    published_at: datetime
    ingested_at: datetime = Field(default_factory=utcnow)
    title: str = ""
    body: str = ""
    language: str = "en"
    engagement: dict[str, int] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    synthetic: bool = False

    @field_validator("published_at", "ingested_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @property
    def text(self) -> str:
        """Title and body joined. Every evidence span is a substring of this string."""
        if self.title and self.body:
            return f"{self.title}\n{self.body}"
        return self.title or self.body


class EntityRef(BaseModel):
    type: Literal["company", "macro"]
    id: str
    name: str
    cik: str | None = None
    sector: str | None = None


class EntityMention(BaseModel):
    """Output of entity linking, mirroring Kensho NERD's recognition and linking scores."""

    entity_id: str
    entity_type: Literal["company", "macro"]
    name: str
    surface: str
    spans: list[tuple[int, int]]
    ner_score: float = Field(ge=0, le=1)
    ned_score: float = Field(ge=0, le=1)
    relevance: int = Field(ge=0, le=100)
    method: str


class EventLabel(BaseModel):
    primary: EventClass
    subtype: str | None = None
    confidence: float = Field(ge=0, le=1)
    labels: list[EventClass] = Field(default_factory=list)
    method: str = "rules-v0"


class Corroboration(BaseModel):
    independent_publishers: int = Field(ge=0)
    source_types: list[SourceType] = Field(default_factory=list)
    authoritative: bool = False
    disputed: bool = False


class Evidence(BaseModel):
    doc_id: str
    publisher: str
    url: str | None = None
    title: str | None = None
    span: str
    weight: float = Field(ge=0)


class Signal(BaseModel):
    """The engine's output contract (schema v1). Published on the bus and served by the API."""

    schema_version: str = SCHEMA_VERSION
    signal_id: str
    grain: Literal["document", "event", "entity"]
    as_of: datetime
    entity: EntityRef
    sentiment_score: float = Field(ge=-1, le=1)
    sentiment_confidence: float = Field(ge=0, le=1)
    event: EventLabel
    impact_score: int = Field(ge=1, le=10)
    p_large_move: float | None = Field(default=None, ge=0, le=1)
    expected_abs_move_bps: float | None = None
    novelty: int = Field(ge=0, le=100)
    relevance: int = Field(ge=0, le=100)
    corroboration: Corroboration
    evidence: list[Evidence] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    synthetic: bool = False

    @field_validator("as_of")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)
