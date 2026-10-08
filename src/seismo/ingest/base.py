"""Adapter interface: an async stream of normalized Documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from seismo.schemas import Document


class Adapter(ABC):
    name: str = "adapter"

    @abstractmethod
    def stream(self) -> AsyncIterator[Document]:
        """Yield documents until the source is exhausted (replay) or cancelled (live)."""
