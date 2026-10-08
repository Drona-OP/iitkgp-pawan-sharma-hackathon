"""Entity-level sentiment: a pluggable model scored only on the sentences that mention each entity.

Backends:
- FinBertBackend: ProsusAI/finbert via transformers (install requirements-ml.txt).
- LexiconBackend: a small, transparent finance lexicon used in CI and when FinBERT is absent.
Day 2 swaps in the fine-tuned, target-masked model behind the same interface.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Protocol

from seismo.schemas import Document, EntityMention

log = logging.getLogger(__name__)

Probs = tuple[float, float, float]  # (negative, neutral, positive)

SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")
TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")


class SentimentBackend(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[Probs]: ...


POSITIVE = frozenset(
    {
        "beat", "beats", "record", "surge", "surges", "surged", "soar", "soars", "soared",
        "rally", "rallies", "rallied", "upgrade", "upgraded", "upgrades", "growth", "gain",
        "gains", "gained", "rise", "rises", "rose", "rising", "jump", "jumps", "jumped",
        "strong", "stronger", "robust", "raises", "raised", "raise", "lifts", "lifted",
        "boost", "boosts", "boosted", "approval", "approved", "approves", "wins", "won", "win",
        "outperform", "outperforms", "upbeat", "optimistic", "recovers", "recovered",
        "recovery", "rebound", "rebounds", "stable", "steady", "exceeds", "exceeded", "tops",
        "topped", "expands", "expansion", "bullish", "overweight", "higher", "buyback",
        "success", "successful", "resilient", "normally",
    }
)
NEGATIVE = frozenset(
    {
        "miss", "misses", "missed", "fall", "falls", "fell", "falling", "drop", "drops",
        "dropped", "decline", "declines", "declined", "slump", "slumps", "plunge", "plunges",
        "plunged", "tumble", "tumbles", "tumbled", "sink", "sinks", "sank", "slide", "slides",
        "slid", "slip", "slips", "slipped", "crash", "crashes", "selloff", "sell-off", "losses",
        "loss", "lose", "loses", "lost", "weak", "weaker", "weakness", "cut", "cuts", "lowered",
        "lowers", "downgrade", "downgraded", "downgrades", "warning", "warns", "warned",
        "fear", "fears", "concern", "concerns", "worries", "worry", "panic", "panicking",
        "probe", "investigation", "lawsuit", "sued", "fined", "fines", "penalty", "recall",
        "recalls", "recalled", "outage", "breach", "hack", "hacked", "bankruptcy", "bankrupt",
        "default", "defaults", "defaulted", "layoffs", "halt", "halts", "halted", "halting",
        "limiting", "grounding", "grounded", "delay", "delays", "delayed", "shortage",
        "sanctions", "tariff", "tariffs", "war", "invasion", "missile", "missiles", "bearish",
        "sell", "underweight", "lower", "worst", "hit", "hits", "hammered", "spooking",
        "spooks", "spooked", "dip", "volatile", "turmoil", "rumors", "rumours",
    }
)
NEGATORS = frozenset(
    {"not", "no", "never", "without", "deny", "denies", "denied", "isn't", "wasn't",
     "aren't", "don't", "doesn't"}
)

# Words that signal acute stress or relief count double: a liquidity crisis is not a soft quarter.
STRONG_NEGATIVE = frozenset(
    {
        "crisis", "collapse", "collapses", "collapsed", "crash", "crashes", "plunge", "plunges",
        "plunged", "bankrupt", "bankruptcy", "insolvent", "insolvency", "default", "defaulted",
        "fraud", "failure", "failed", "fails", "panic", "rout", "turmoil", "contagion",
        "receivership", "halt", "halts", "halted", "halting", "bloodbath", "brutal", "crushed",
        "wiped", "tumble", "tumbles", "tumbled", "worst", "selloff", "sell-off", "run",
    }
)
EXTRA_NEGATIVE = frozenset(
    {
        "doubts", "doubt", "worried", "nervous", "uncertain", "pressure", "rattles", "rattled",
        "hurt", "hurts", "dilution", "retaliates", "retaliatory", "escalates", "escalated",
        "escalating", "recession", "slowdown", "slows", "slower", "slides", "sliding", "drags",
        "dragging", "fallout", "outflows", "ugly", "losing", "rattle", "threat", "threatens",
        "risk", "risks", "spook", "scramble", "fled", "flee", "flight", "suspended", "suspends",
        "downturn", "shock", "slashes", "slashed",
    }
)
EXTRA_POSITIVE = frozenset(
    {
        "soar", "soaring", "rallied", "ripping", "jumped", "best", "recover", "protected",
        "backstop", "inflows", "rebounds", "rebounded", "beats", "upbeat", "reassures",
        "reassured", "accelerates", "accelerating", "record-high", "approval", "eases", "eased",
        "easing", "pause", "relief",
    }
)
POSITIVE = POSITIVE | EXTRA_POSITIVE
NEGATIVE = (NEGATIVE | EXTRA_NEGATIVE | STRONG_NEGATIVE) - {"pause"}
STRONG_POSITIVE = frozenset({"soar", "soars", "soared", "surge", "surges", "surged", "record", "best", "rally", "rallied"})


INTENSIFIERS = frozenset({"sharp", "sharply", "steep", "steeply", "heavy", "heavily", "massive",
                          "biggest", "record", "deep", "deepest", "severe", "brutal"})
QUESTION_WORDS = frozenset({"question", "questions", "questioned", "challenges", "challenge", "selling"})
NEUTRALISED_PHRASES = re.compile(r"\bgains? traction\b|\bpaused?\b.{0,20}\bfor\b", re.IGNORECASE)
MOVE_RE = re.compile(
    r"(?P<sign>[-+\u2212])\s?\d+(?:\.\d+)?\s?%"
    r"|\b(?P<down>down|fell|falls?|drops?|dropped|slid|slides|sank|sinks|lost|loses|lose|plunged|plunges|tumbled|tumbles|declined|declines)\b(?:\s+(?:about|more than|nearly|almost|over|roughly|some))?\s+\d+(?:\.\d+)?\s?%"
    r"|\b(?P<up>up|rose|rises|jumped|jumps|gained|gains|climbed|climbs|soared|soars|surged|surges|rallied|rallies)\b(?:\s+(?:about|more than|nearly|almost|over|roughly|some))?\s+\d+(?:\.\d+)?\s?%",
    re.IGNORECASE,
)


class LexiconBackend:
    """Transparent fallback for CI and offline runs. FinBERT replaces it when installed.

    A finance word list (positive, negative, and doubled weights for acute-stress words), a
    three-token negation window, intensifiers ("sharp losses"), and signed percentage moves
    ("-12%", "fell 17%") read as strong polarity.
    """

    name = "lexicon-v2"

    def predict(self, texts: list[str]) -> list[Probs]:
        return [self._score(t) for t in texts]

    @staticmethod
    def _score(text: str) -> Probs:
        pos = neg = 0.0
        for m in MOVE_RE.finditer(text):
            if m.group("sign"):
                if m.group("sign") in "-\u2212":
                    neg += 2.0
                else:
                    pos += 2.0
            elif m.group("down"):
                neg += 2.0
            elif m.group("up"):
                pos += 2.0
        cleaned = MOVE_RE.sub(" ", NEUTRALISED_PHRASES.sub(" ", text))
        tokens = TOKEN_RE.findall(cleaned.lower())
        for i, tok in enumerate(tokens):
            polarity = 1 if tok in POSITIVE else -1 if tok in NEGATIVE or tok in QUESTION_WORDS else 0
            if polarity == 0:
                continue
            weight = 2.0 if tok in STRONG_NEGATIVE or tok in STRONG_POSITIVE else 1.0
            if any(t in INTENSIFIERS for t in tokens[max(0, i - 2):i]):
                weight *= 1.5
            if any(t in NEGATORS for t in tokens[max(0, i - 3):i]):
                polarity = -polarity
            if polarity > 0:
                pos += weight
            else:
                neg += weight
        total = pos + neg
        if total == 0:
            return (0.1, 0.8, 0.1)
        balance = (pos - neg) / total
        strength = 0.95 * min(1.0, total / 3.0)
        p_pos = 0.5 * strength * (1 + balance)
        p_neg = 0.5 * strength * (1 - balance)
        return (p_neg, 1.0 - p_pos - p_neg, p_pos)


class FinBertBackend:
    """ProsusAI/finbert through the transformers pipeline (CPU by default)."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1) -> None:
        from transformers import pipeline  # lazy: heavy import, optional dependency

        self._pipe = pipeline(
            "text-classification", model=model_name, top_k=None, truncation=True, device=device
        )
        self.name = f"finbert:{model_name}"

    def predict(self, texts: list[str]) -> list[Probs]:
        if not texts:
            return []
        outputs = self._pipe(texts, batch_size=16)
        probs: list[Probs] = []
        for out in outputs:
            scores = {d["label"].lower(): float(d["score"]) for d in out}
            probs.append(
                (scores.get("negative", 0.0), scores.get("neutral", 0.0), scores.get("positive", 0.0))
            )
        return probs


def make_backend(kind: str = "auto", model_name: str = "ProsusAI/finbert") -> SentimentBackend:
    kind = (kind or "auto").lower()
    if kind == "lexicon":
        return LexiconBackend()
    if kind == "finbert":
        return FinBertBackend(model_name)
    try:
        return FinBertBackend(model_name)
    except Exception as exc:  # noqa: BLE001 - any import/download failure falls back
        log.warning("FinBERT unavailable (%s); using the lexicon fallback.", exc.__class__.__name__)
        return LexiconBackend()


def entity_windows(doc: Document, mention: EntityMention, limit: int = 3) -> list[tuple[int, int]]:
    """Character ranges of the sentences that mention the entity (title counts as a sentence)."""
    text = doc.text
    sentences = [(m.start(), m.end()) for m in SENTENCE_RE.finditer(text) if m.group().strip()]
    starts = [s for s, _ in mention.spans]
    chosen = [(s, e) for s, e in sentences if any(s <= p < e for p in starts)]
    if not chosen:
        chosen = sentences[:1] or [(0, len(text))]
    return chosen[:limit]


def average(probs: list[Probs]) -> Probs:
    n = max(1, len(probs))
    return (
        sum(p[0] for p in probs) / n,
        sum(p[1] for p in probs) / n,
        sum(p[2] for p in probs) / n,
    )


def confidence(p: Probs) -> float:
    """1 minus normalized entropy: 1 for a certain prediction, 0 for a uniform one."""
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(3)))
