"""SQLite store in WAL mode, so the dashboard can read while the pipeline writes."""

from __future__ import annotations

import sqlite3
import statistics
import threading
from datetime import UTC, datetime
from pathlib import Path

from seismo.schemas import Document, Signal

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    published_at TEXT NOT NULL,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT UNIQUE NOT NULL,
    grain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    impact INTEGER NOT NULL,
    sentiment REAL NOT NULL,
    event_primary TEXT NOT NULL,
    engine_ms REAL,
    json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_signals_grain_entity ON signals (grain, entity_id);
CREATE INDEX IF NOT EXISTS ix_signals_as_of ON signals (as_of);
"""


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM signals")
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()

    def put_document(self, doc: Document) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, source, source_type, published_at, json) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc.doc_id, doc.source, doc.source_type.value, _iso(doc.published_at), doc.model_dump_json()),
            )
            self._conn.commit()

    def put_signal(self, sig: Signal) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO signals (signal_id, grain, entity_id, as_of, impact, sentiment, "
                "event_primary, engine_ms, json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(signal_id) DO UPDATE SET impact=excluded.impact, "
                "sentiment=excluded.sentiment, event_primary=excluded.event_primary, "
                "engine_ms=excluded.engine_ms, json=excluded.json",
                (
                    sig.signal_id, sig.grain, sig.entity.id, _iso(sig.as_of), sig.impact_score,
                    sig.sentiment_score, sig.event.primary.value, sig.latency_ms.get("engine"),
                    sig.model_dump_json(),
                ),
            )
            self._conn.commit()

    def document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute("SELECT json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return Document.model_validate_json(row[0]) if row else None

    def signals(
        self,
        grain: str | None = None,
        entity: str | None = None,
        min_impact: int | None = None,
        limit: int = 200,
    ) -> list[Signal]:
        """Most recent first (by event time, then insertion order)."""
        clauses, params = [], []
        if grain:
            clauses.append("grain = ?")
            params.append(grain)
        if entity:
            clauses.append("entity_id = ?")
            params.append(entity)
        if min_impact is not None:
            clauses.append("impact >= ?")
            params.append(min_impact)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT json FROM signals {where} ORDER BY as_of DESC, seq DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [Signal.model_validate_json(r[0]) for r in rows]

    def signals_after(self, seq: int, limit: int = 500) -> list[tuple[int, Signal]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, json FROM signals WHERE seq > ? ORDER BY seq LIMIT ?", (seq, limit)
            ).fetchall()
        return [(r[0], Signal.model_validate_json(r[1])) for r in rows]

    def max_seq(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM signals").fetchone()
        return int(row[0])

    def latest_entities(self) -> list[Signal]:
        """The newest entity-grain signal per entity."""
        sql = (
            "SELECT s.json FROM signals s JOIN ("
            "  SELECT entity_id, MAX(seq) AS seq FROM signals WHERE grain = 'entity' GROUP BY entity_id"
            ") latest ON latest.seq = s.seq ORDER BY s.impact DESC, ABS(s.sentiment) DESC"
        )
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [Signal.model_validate_json(r[0]) for r in rows]

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            sigs = self._conn.execute(
                "SELECT COUNT(*) FROM signals WHERE grain = 'document'"
            ).fetchone()[0]
            severe = self._conn.execute(
                "SELECT COUNT(*) FROM signals WHERE grain = 'document' AND impact >= 8"
            ).fetchone()[0]
            lat = [
                r[0]
                for r in self._conn.execute(
                    "SELECT engine_ms FROM signals WHERE grain = 'document' AND engine_ms IS NOT NULL "
                    "ORDER BY seq DESC LIMIT 500"
                ).fetchall()
            ]
        return {
            "documents": int(docs),
            "signals": int(sigs),
            "severe": int(severe),
            "median_engine_ms": round(statistics.median(lat), 2) if lat else 0.0,
        }
