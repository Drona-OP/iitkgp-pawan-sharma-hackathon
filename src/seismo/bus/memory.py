"""In-memory fan-out bus on asyncio queues (single process, used for demo, replay and tests)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from seismo.bus.base import END, Bus, Subscription


class InMemoryBus(Bus):
    def __init__(self, maxsize: int = 10_000) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._maxsize = maxsize

    async def publish(self, topic: str, key: str | None, value: Any) -> None:
        for queue in list(self._subs[topic]):
            await queue.put((key, value))

    def subscribe(self, topic: str) -> Subscription:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs[topic].append(queue)
        return Subscription(queue)

    async def end(self, topic: str) -> None:
        for queue in list(self._subs[topic]):
            await queue.put(END)
