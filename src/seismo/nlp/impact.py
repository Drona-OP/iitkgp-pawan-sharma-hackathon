"""Impact: how much an event could move its asset if it is true, on a 1-10 scale.

Impact answers "how big if true"; the trigger gate (signals/gate.py) answers "is it true". Keeping
the two apart lets a risk manager see a dangerous rumour as impact 9 / REVIEW instead of having
credibility silently shrink it.

Two scorers share one feature set:
- ``PriorImpact`` (default): a transparent logistic score with hand-set weights on event class,
  sentiment strength, diffusion (independent publishers, social reach), authority, novelty and
  relevance. It emits no probability, because none of it is fitted.
- ``CalibratedImpact``: the event-study model trained by ``seismo.eval.impact_study`` on SEC 8-K
  events and next-day abnormal returns (|SCAR| > 2), isotonic-calibrated. It returns a real
  probability of a 2-sigma move; the diffusion terms are added as a documented prior on the logit.

impact = min(10, max(1, ceil(10 * p))), so "impact 8" means roughly a 70-80% chance of a 2-sigma move
once the calibrated model is loaded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from seismo.schemas import EventClass

VERSION = "logit-prior-v1"

E = EventClass
CLASS_PRIOR: dict[EventClass, float] = {
    E.CREDIT_EVENT: 1.4,
    E.GEOPOLITICAL: 1.1,
    E.MACROECONOMIC: 0.8,
    E.EARNINGS_GUIDANCE: 0.6,
    E.OPERATIONAL_ESG: 0.5,
    E.MA_CORPORATE_ACTION: 0.5,
    E.LEGAL_REGULATORY: 0.4,
    E.PRODUCT_STRATEGY: 0.4,
    E.MANAGEMENT_GOVERNANCE: 0.2,
    E.OTHER: -1.0,
}
INTERCEPT = -3.8
W_SENTIMENT = 1.8
W_PUBLISHERS = 0.45   # per doubling of independent publishers
W_SOCIAL = 0.35       # per doubling of distinct social accounts, capped at 32 accounts
W_AUTHORITY = 0.3
W_NOVELTY = 0.6
W_RELEVANCE = 0.5
SOCIAL_CAP = 32
MIN_SUPPORT = 30   # training events a class needs before its calibrated rate is used


@dataclass(frozen=True)
class ImpactFeatures:
    event: EventClass
    sentiment: float
    publishers: int = 1
    social_authors: int = 0
    authoritative: bool = False
    novelty: int = 100
    relevance: int = 100
    subtype: str | None = None


def to_impact(p: float) -> int:
    return max(1, min(10, math.ceil(10 * p - 1e-9)))


def diffusion_logit(f: ImpactFeatures) -> float:
    return (
        W_PUBLISHERS * math.log2(1 + max(0, f.publishers))
        + W_SOCIAL * math.log2(1 + min(SOCIAL_CAP, max(0, f.social_authors)))
        + W_NOVELTY * f.novelty / 100
        + W_RELEVANCE * f.relevance / 100
    )


class PriorImpact:
    name = VERSION
    calibrated = False

    def logit(self, f: ImpactFeatures) -> float:
        return (
            INTERCEPT
            + CLASS_PRIOR[f.event]
            + W_SENTIMENT * min(1.0, abs(f.sentiment))
            + W_AUTHORITY * float(f.authoritative)
            + diffusion_logit(f)
        )

    def score(self, f: ImpactFeatures) -> tuple[int, float]:
        p = 1 / (1 + math.exp(-self.logit(f)))
        return to_impact(p), p

    def score_ex(self, f: ImpactFeatures) -> tuple[int, float, bool]:
        """Impact, score and whether the score is a calibrated probability."""
        impact, p = self.score(f)
        return impact, p, False


class CalibratedImpact(PriorImpact):
    """Loads models/impact_calibrated.json written by ``python -m seismo.eval.impact_study``.

    The file holds a per-(class) base logit learned from the 8-K event study, the fitted
    sentiment-free coefficients, and an isotonic map from raw score to probability.
    """

    calibrated = True

    def __init__(self, path: str | Path) -> None:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name = spec.get("version", "event-study")
        self.class_logit = {E(k): float(v) for k, v in spec["class_logit"].items()}
        self.default_logit = float(spec.get("default_logit", -2.0))
        self.iso_x = [float(x) for x in spec["isotonic"]["x"]]
        self.iso_y = [float(y) for y in spec["isotonic"]["y"]]
        self.diffusion_centre = float(spec.get("diffusion_centre", 1.6))
        support = spec.get("class_support", {})
        self.supported = {E(k) for k, n in support.items() if int(n) >= MIN_SUPPORT} if support else set(self.class_logit)
        self.prior = PriorImpact()

    def _iso(self, z: float) -> float:
        xs, ys = self.iso_x, self.iso_y
        if z <= xs[0]:
            return ys[0]
        if z >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if z <= xs[i]:
                t = (z - xs[i - 1]) / max(1e-12, xs[i] - xs[i - 1])
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]

    def logit(self, f: ImpactFeatures) -> float:
        base = self.class_logit.get(f.event, self.default_logit)
        # Sentiment strength and diffusion are not in the 8-K training data; they enter as a
        # centred prior so an average story keeps the event-study base rate.
        return (
            base
            + W_SENTIMENT * (min(1.0, abs(f.sentiment)) - 0.5)
            + (diffusion_logit(f) - self.diffusion_centre)
        )

    def score(self, f: ImpactFeatures) -> tuple[int, float]:
        impact, p, _ = self.score_ex(f)
        return impact, p

    def score_ex(self, f: ImpactFeatures) -> tuple[int, float, bool]:
        """Classes the 8-K study never saw (macro, geopolitical, product news) keep the prior."""
        if f.event not in self.supported:
            impact, p = self.prior.score(f)
            return impact, p, False
        p = self._iso(self.logit(f))
        return to_impact(p), p, True


def load_impact_model(path: str | Path | None) -> PriorImpact:
    if path and Path(path).exists():
        try:
            return CalibratedImpact(path)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return PriorImpact()


def heuristic_impact(
    primary: EventClass, sentiment: float, credibility: float, relevance: int, novelty: int
) -> tuple[int, float]:
    """Document-grain score kept for backwards compatibility: one publisher, no social reach."""
    del credibility  # credibility belongs to the gate, not to impact
    return PriorImpact().score(
        ImpactFeatures(primary, sentiment, publishers=1, novelty=novelty, relevance=relevance)
    )
