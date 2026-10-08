"""Small pricing toolkit: bond duration and convexity, swap DV01, CDS CS01, Black-Scholes greeks.

Everything is a sensitivity, because a stress test revalues a book with a Taylor expansion:

    bonds:    dP/P ~ -D_mod dy + 1/2 C dy^2,   dy = d(rate at tenor) + d(spread for rating)
    swaps:    dV   ~ DV01 x d(rate in bp)      (sign from pay or receive fixed)
    CDS:      dV   ~ CS01 x d(spread in bp)    (sign from bought or sold protection)
    options:  dV   ~ delta dS + 1/2 gamma dS^2 + vega d(sigma)
"""

from __future__ import annotations

import math
from statistics import NormalDist

N = NormalDist()


def bond_risk(coupon: float, maturity: float, ytm: float, freq: int = 2) -> tuple[float, float, float]:
    """Price per 100, modified duration and convexity of a plain fixed-coupon bond."""
    n = max(1, round(maturity * freq))
    c = 100 * coupon / freq
    y = ytm / freq
    price = mac = conv = 0.0
    for k in range(1, n + 1):
        cf = c + (100 if k == n else 0)
        df = (1 + y) ** -k
        pv = cf * df
        t = k / freq
        price += pv
        mac += t * pv
        conv += cf * k * (k + 1) * df / ((1 + y) ** 2 * freq**2)
    mac /= price
    mod = mac / (1 + y)
    return price, mod, conv / price


def swap_dv01(notional: float, maturity: float, rate: float, receive_fixed: bool) -> float:
    """P&L for a +1 bp parallel rise: receivers lose, payers gain."""
    _, mod, _ = bond_risk(rate, maturity, rate)  # par bond: the fixed leg's rate sensitivity
    dv01 = notional * mod * 1e-4
    return -dv01 if receive_fixed else dv01


def cds_cs01(notional: float, maturity: float, spread: float, rate: float, bought: bool) -> float:
    """P&L for a +1 bp spread widening: protection buyers gain."""
    hazard = spread / 0.6
    k = rate + hazard
    rpv01 = (1 - math.exp(-k * maturity)) / k if k > 0 else maturity
    cs01 = notional * rpv01 * 1e-4
    return cs01 if bought else -cs01


def bs_greeks(spot: float, strike: float, years: float, vol: float, rate: float, call: bool) -> dict[str, float]:
    years = max(years, 1 / 365)
    vol = max(vol, 0.01)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    if call:
        price = spot * N.cdf(d1) - strike * math.exp(-rate * years) * N.cdf(d2)
        delta = N.cdf(d1)
    else:
        price = strike * math.exp(-rate * years) * N.cdf(-d2) - spot * N.cdf(-d1)
        delta = N.cdf(d1) - 1
    gamma = pdf / (spot * vol * math.sqrt(years))
    vega = spot * pdf * math.sqrt(years)  # per 1.00 of vol; x0.01 per vol point
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega}
