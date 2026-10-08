"""Bus interface. Messages are (key, value) pairs; keys are entity ids so per-entity order holds."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

TOPIC_DOCS = "docs.raw"
TOPIC_SIGNALS = "signals.v1"

END: Any = object()  # end-of-stream marker


class Subscription:
    """Async iterator over (key, value) messages; stops at END."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> tuple[str | None, Any]:
        item = await self._queue.get()
        if item is END:
            raise StopAsyncIteration
        return item


class Bus(ABC):
    @abstractmethod
    async def publish(self, topic: str, key: str | None, value: Any) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str) -> Subscription: ...

    @abstractmethod
    async def end(self, topic: str) -> None:
        """Signal end-of-stream to every subscriber of a topic."""
