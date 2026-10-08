"""MinHash signatures with banded LSH, in pure Python so replays stay deterministic and dependency-free.

A signature estimates the Jaccard similarity of two documents' word shingles; banding turns that
into a fast candidate lookup (two texts collide in at least one band with high probability when
their Jaccard similarity is above about (1/bands)^(1/rows)).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Hashable, Iterable

TOKEN_RE = re.compile(r"[a-z0-9$][a-z0-9'$-]*")
STOPWORDS = frozenset(
    """a an the of to in on for and or as at by with its is are was were be been after over from
    this that it into than then so but not no if about amid says said say new more most just very
    can could will would may might has have had do does did up down out off per via vs""".split()
)

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def content_words(text: str) -> frozenset[str]:
    """Distinct content words: a crude but deterministic paraphrase key."""
    return frozenset(t for t in tokens(text) if len(t) > 2 and not t.isdigit())


def shingles(text: str, k: int = 3) -> set[str]:
    toks = tokens(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _h32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")


def _params(n: int, seed: int = 7) -> list[tuple[int, int]]:
    out = []
    for i in range(n):
        d = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=16).digest()
        a = int.from_bytes(d[:8], "big") % _MERSENNE or 1
        b = int.from_bytes(d[8:], "big") % _MERSENNE
        out.append((a, b))
    return out


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 7) -> None:
        self.num_perm = num_perm
        self._ab = _params(num_perm, seed)

    def signature(self, items: Iterable[str]) -> tuple[int, ...]:
        hashed = [_h32(x) for x in items]
        if not hashed:
            return tuple([_MAX_HASH] * self.num_perm)
        return tuple(min(((a * h + b) % _MERSENNE) & _MAX_HASH for h in hashed) for a, b in self._ab)

    @staticmethod
    def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        if not a or len(a) != len(b):
            return 0.0
        return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LSHIndex:
    """Banded LSH over MinHash signatures. Keys are any hashable (here: cluster ids)."""

    def __init__(self, num_perm: int = 64, bands: int = 16) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.bands = bands
        self.rows = num_perm // bands
        self._buckets: dict[tuple[int, tuple[int, ...]], set[Hashable]] = defaultdict(set)

    def _bands(self, sig: tuple[int, ...]) -> Iterable[tuple[int, tuple[int, ...]]]:
        for b in range(self.bands):
            yield b, sig[b * self.rows:(b + 1) * self.rows]

    def add(self, key: Hashable, sig: tuple[int, ...]) -> None:
        for band in self._bands(sig):
            self._buckets[band].add(key)

    def candidates(self, sig: tuple[int, ...]) -> set[Hashable]:
        found: set[Hashable] = set()
        for band in self._bands(sig):
            found |= self._buckets.get(band, set())
        return found
