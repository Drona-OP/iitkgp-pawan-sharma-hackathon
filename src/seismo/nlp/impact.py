"""Impact v0: a transparent heuristic, replaced on Day 3 by the event-study model.

p = class severity x sentiment strength x credibility x relevance x novelty, mapped to 1-10.
It is deliberately labelled heuristic and emits no probability; the calibrated model will.
"""

from __future__ import annotations

import math

from seismo.schemas import EventClass

VERSION = "heuristic-v0"

E = EventClass
CLASS_SEVERITY: dict[EventClass, float] = {
    E.CREDIT_EVENT: 1.0,
    E.GEOPOLITICAL: 0.8,
    E.MACROECONOMIC: 0.7,
    E.EARNINGS_GUIDANCE: 0.65,
    E.MA_CORPORATE_ACTION: 0.6,
    E.LEGAL_REGULATORY: 0.55,
    E.OPERATIONAL_ESG: 0.55,
    E.PRODUCT_STRATEGY: 0.5,
    E.MANAGEMENT_GOVERNANCE: 0.45,
    E.OTHER: 0.2,
}


def heuristic_impact(
    primary: EventClass, sentiment: float, credibility: float, relevance: int, novelty: int
) -> tuple[int, float]:
    p = (
        CLASS_SEVERITY[primary]
        * (0.35 + 0.65 * min(1.0, abs(sentiment)))
        * (0.5 + 0.5 * credibility)
        * (0.6 + 0.4 * relevance / 100)
        * (0.5 + 0.5 * novelty / 100)
    )
    return max(1, min(10, math.ceil(10 * p))), p
