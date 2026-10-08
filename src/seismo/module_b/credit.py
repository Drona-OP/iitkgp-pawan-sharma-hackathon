"""Credit risk maths: rating scale, PDs, the Vasicek/Basel IRB stressed PD, staging and ECL.

PDs. One-year PDs by notch are a smoothed approximation of S&P Global Ratings' published
long-run average one-year corporate default rates (annual default and transition studies),
floored at the Basel 3 basis-point PD floor. They are a public, sponsor-native calibration; the
by-notch smoothing is ours.

Stressed PD (the Vasicek single-factor model behind the Basel IRB formula):

    PD(Z) = Phi( (Phi^-1(PD) - sqrt(rho) Z) / sqrt(1 - rho) ),  Z < 0 is a bad state of the world
    rho   = 0.12 (1 - e^{-50 PD}) / (1 - e^{-50}) + 0.24 (1 - (1 - e^{-50 PD}) / (1 - e^{-50}))

Rating migration is damped: a systematic shock moves an obligor by
    notches = round(k x log2(PD(Z) / PD)),  k = 0.75 (migration elasticity, an assumption),
because the Vasicek PD(Z) is the expected default rate of a bucket in a bad state, not every
obligor's new rating. Idiosyncratic notches from the event are added on top.

Staging follows IFRS 9 / RBI ECL logic with illustrative proxies (not RBI's exact rules):
Stage 2 on a significant increase in credit risk (3+ notches below origination, or BB- or worse),
Stage 3 on default. ECL: Stage 1 = 12-month PD x LGD x EAD; Stage 2 = lifetime PD x LGD x EAD;
Stage 3 = LGD x EAD.
"""

from __future__ import annotations

import math
from statistics import NormalDist

N = NormalDist()

SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
         "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "D"]
NOTCH = {r: i for i, r in enumerate(SCALE)}

PD_FLOOR = 0.0003
# One-year PD (as a fraction) by notch. Smoothed from S&P long-run averages; D is default.
PD_1Y: dict[str, float] = {
    "AAA": 0.0003, "AA+": 0.0003, "AA": 0.0003, "AA-": 0.0004, "A+": 0.0005, "A": 0.0006,
    "A-": 0.0007, "BBB+": 0.0010, "BBB": 0.0015, "BBB-": 0.0025, "BB+": 0.0035, "BB": 0.0055,
    "BB-": 0.0100, "B+": 0.0190, "B": 0.0300, "B-": 0.0600, "CCC": 0.2500, "D": 1.0,
}

# Basel III standardised corporate risk weights by external rating bucket.
def risk_weight(rating: str | None, sovereign: bool = False) -> float:
    if sovereign:
        return 0.0
    if rating is None or rating not in NOTCH:
        return 1.0
    n = NOTCH[rating]
    if n <= NOTCH["AA-"]:
        return 0.20
    if n <= NOTCH["A-"]:
        return 0.50
    if n <= NOTCH["BBB-"]:
        return 0.75
    if n <= NOTCH["BB-"]:
        return 1.00
    return 1.50


def notch_down(rating: str, notches: int) -> str:
    n = min(NOTCH[rating] + max(0, notches), NOTCH["D"])
    return SCALE[n]


def rating_for_pd(pd: float) -> str:
    """The best rating whose one-year PD is at least the given PD (implied rating)."""
    for r in SCALE[:-1]:
        if PD_1Y[r] >= pd - 1e-12:
            return r
    return "CCC" if pd < 1.0 else "D"


def basel_rho(pd: float) -> float:
    k = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    return 0.12 * k + 0.24 * (1 - k)


def vasicek_pd(pd: float, z: float, rho: float | None = None) -> float:
    """Conditional PD given the systematic factor z (z = -2 is roughly a 1-in-44 bad year)."""
    if pd >= 1.0:
        return 1.0
    pd = max(pd, PD_FLOOR)
    rho = basel_rho(pd) if rho is None else rho
    return N.cdf((N.inv_cdf(pd) - math.sqrt(rho) * z) / math.sqrt(1 - rho))


MIGRATION_ELASTICITY = 0.75


def systematic_notches(pd: float, z: float) -> int:
    if z >= 0 or pd >= 1.0:
        return 0
    ratio = vasicek_pd(pd, z) / max(pd, PD_FLOOR)
    return max(0, round(MIGRATION_ELASTICITY * math.log2(ratio)))


def lifetime_pd(pd_1y: float, years: float, pd_later: float | None = None) -> float:
    """Cumulative PD over the remaining life: stressed first year, then the through-the-cycle PD."""
    years = max(1.0, years)
    later = pd_1y if pd_later is None else pd_later
    return 1 - (1 - min(pd_1y, 1.0)) * (1 - min(later, 1.0)) ** (years - 1)


def stage(origination: str, current: str, defaulted: bool = False) -> int:
    if defaulted or current == "D":
        return 3
    if NOTCH[current] - NOTCH[origination] >= 3 or NOTCH[current] >= NOTCH["BB-"]:
        return 2
    return 1


def ecl(stage_: int, pd_1y: float, lgd: float, ead: float, years: float, pd_later: float | None = None) -> float:
    if stage_ == 3:
        return lgd * ead
    if stage_ == 2:
        return lifetime_pd(pd_1y, years, pd_later) * lgd * ead
    return min(pd_1y, 1.0) * lgd * ead
