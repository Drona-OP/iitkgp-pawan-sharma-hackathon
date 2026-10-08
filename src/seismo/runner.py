"""Wires adapters -> bus (docs.raw) -> engine -> bus (signals.v1) -> store and consumers.

Same code live and replay. Consumers are anything with ``on_signal(signal)``: the trigger gate,
Module A (tactical rebalancer) and Module B (stress test) all subscribe to the one signal bus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from seismo.bus import TOPIC_DOCS, TOPIC_SIGNALS, Bus, InMemoryBus
from seismo.engine import Engine
from seismo.ingest.base import Adapter
from seismo.ingest.replay import Recorder
from seismo.schemas import Signal
from seismo.signals.aggregator import EntityAggregator
from seismo.store.sqlite import SQLiteStore

log = logging.getLogger(__name__)


class Consumer(Protocol):
    def on_signal(self, sig: Signal) -> object: ...


async def run_pipeline(
    adapters: list[Adapter],
    engine: Engine,
    aggregator: EntityAggregator,
    store: SQLiteStore,
    bus: Bus | None = None,
    recorder: Recorder | None = None,
    stop_after_seconds: float | None = None,
    consumers: list[Consumer] | None = None,
) -> dict[str, int]:
    bus = bus or InMemoryBus()
    docs = bus.subscribe(TOPIC_DOCS)
    signals = bus.subscribe(TOPIC_SIGNALS)
    counts = {"documents": 0, "signals": 0, "events": 0}
    consumers = list(consumers or [])

    async def produce(adapter: Adapter) -> None:
        try:
            async for doc in adapter.stream():
                await bus.publish(TOPIC_DOCS, doc.doc_id, doc)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one failing source must not stop the others
            log.exception("Adapter %s failed", adapter.name)

    async def process() -> None:
        async for _, doc in docs:
            t0 = time.perf_counter()
            store.put_document(doc)
            if recorder is not None:
                recorder.write(doc)
            counts["documents"] += 1
            result = await asyncio.to_thread(engine.process_full, doc)
            pipeline_ms = round((time.perf_counter() - t0) * 1000, 3)
            for sig in result.doc_signals:
                sig.latency_ms["pipeline"] = pipeline_ms
                await bus.publish(TOPIC_SIGNALS, sig.entity.id, sig)
            for sig in result.event_signals:
                sig.latency_ms["pipeline"] = pipeline_ms
                await bus.publish(TOPIC_SIGNALS, sig.entity.id, sig)
                await bus.publish(TOPIC_SIGNALS, sig.entity.id, aggregator.update(sig))
        await bus.end(TOPIC_SIGNALS)

    async def sink() -> None:
        async for _, sig in signals:
            store.put_signal(sig)
            if sig.grain == "document":
                counts["signals"] += 1
            elif sig.grain == "event":
                counts["events"] += 1
            for consumer in consumers:
                try:
                    consumer.on_signal(sig)
                except Exception:  # noqa: BLE001 - a consumer bug must not stop the engine
                    log.exception("Consumer %s failed", type(consumer).__name__)

    producers = [asyncio.create_task(produce(a), name=f"adapter:{a.name}") for a in adapters]
    processor = asyncio.create_task(process(), name="engine")
    writer = asyncio.create_task(sink(), name="store")
    try:
        if stop_after_seconds:
            _, pending = await asyncio.wait(producers, timeout=stop_after_seconds)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        else:
            await asyncio.gather(*producers)
    finally:
        await bus.end(TOPIC_DOCS)
        await processor
        await writer
        if recorder is not None:
            recorder.close()
    return counts
