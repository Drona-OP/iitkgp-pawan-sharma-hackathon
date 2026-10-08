"""Impact calibrated on real market reactions: an event study on SEC 8-K filings.

    python -m seismo.eval.impact_study      # needs data/market/edgar_8k.csv and prices_daily.csv

1. Each 8-K for the 20 universe names gets an event class from its item codes (1.03 bankruptcy,
   1.05 cyber incident, 2.02 results, 4.02 restatement, 5.02 officer change, ...).
2. Event day t0 is the first session whose close comes after the SEC acceptance time
   (acceptance after 16:00 ET rolls to the next session). Filings on the same ticker-day merge.
3. Market model on days t0-250 .. t0-30:  r_i = a + b r_SPY + e, residual volatility s_e.
4. SCAR = (AR_0 + AR_1) / (s_e sqrt 2); label y = 1{|SCAR| > 2}: a two-sigma reaction.
5. A logistic model on event class, item count, timing (pre-market / intraday / after hours),
   trailing 20-day volatility relative to the market model, and the VIX level, trained on
   2015-2020, isotonic-calibrated on 2021, tested on 2022-2023, then checked on 2024-2025.
6. Benchmarks: the base rate, and the hand-set 8-K severity ranking a typical rules engine uses.

Writes models/impact_calibrated.json (loaded by the engine), docs/results/impact_report.json and
a reliability table. SEC acceptance times are US Eastern.
"""

from __future__ import annotations

import json
import math
from datetime import time as dtime

import numpy as np
import pandas as pd

from seismo.config import load_settings
from seismo.nlp.events import ITEM_MAP
from seismo.schemas import EventClass

TRAIN = ("2015-01-01", "2020-12-31")
VALID = ("2021-01-01", "2021-12-31")
TEST = ("2022-01-01", "2023-12-31")
POST = ("2024-01-01", "2026-12-31")
CLASSES = [c.value for c in EventClass]


def classify_items(items: str) -> tuple[str, str | None, int, int]:
    codes = [c.strip() for c in str(items or "").split(",") if c.strip()]
    mapped = [ITEM_MAP[c] for c in codes if c in ITEM_MAP]
    if not mapped:
        return EventClass.OTHER.value, None, 0, len(codes)
    cls, sub, rank = max(mapped, key=lambda m: m[2])
    return cls.value, sub, rank, len(codes)


def timing_bucket(ts: pd.Timestamp) -> str:
    t = ts.time()
    if t < dtime(9, 30):
        return "pre"
    if t < dtime(16, 0):
        return "intraday"
    return "after"


def build_events(edgar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker x event day with SCAR and features."""
    px = prices.sort_index()
    rets = np.log(px).diff()
    spy = rets["SPY"]
    days = px.index
    vix = px["VIX"] if "VIX" in px.columns else pd.Series(np.nan, index=days)

    rows = []
    for _, f in edgar.iterrows():
        ticker = f["ticker"]
        if ticker not in rets.columns:
            continue
        raw = str(f.get("acceptance_datetime") or f["filing_date"])
        ts = pd.Timestamp(raw.replace("Z", "").replace(".000", "")[:19])  # SEC times are Eastern
        day = ts.normalize()
        pos = days.searchsorted(day)
        if pos >= len(days):
            continue
        if days[pos] == day and timing_bucket(ts) == "after":
            pos += 1
        if pos + 1 >= len(days) or pos < 260:
            continue
        cls, sub, rank, n_items = classify_items(f.get("items", ""))
        rows.append({"ticker": ticker, "pos": pos, "t0": days[pos], "cls": cls, "sub": sub, "rank": rank,
                     "n_items": n_items, "timing": timing_bucket(ts), "items": f.get("items", "")})
    if not rows:
        return pd.DataFrame()
    ev = pd.DataFrame(rows)
    # Merge same ticker-day: keep the most severe class, count all items.
    ev = ev.sort_values("rank", ascending=False).groupby(["ticker", "pos"], as_index=False).agg(
        t0=("t0", "first"), cls=("cls", "first"), sub=("sub", "first"), rank=("rank", "max"),
        n_items=("n_items", "sum"), timing=("timing", "first"), filings=("cls", "size"))

    out = []
    for _, e in ev.iterrows():
        r = rets[e["ticker"]]
        p = int(e["pos"])
        est = slice(p - 250, p - 30)
        y, x = r.iloc[est].to_numpy(), spy.iloc[est].to_numpy()
        ok = ~(np.isnan(y) | np.isnan(x))
        if ok.sum() < 120:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        resid = y[ok] - (a + b * x[ok])
        s_e = float(np.std(resid, ddof=2))
        ar0 = r.iloc[p] - (a + b * spy.iloc[p])
        ar1 = r.iloc[p + 1] - (a + b * spy.iloc[p + 1])
        if np.isnan(ar0) or np.isnan(ar1) or s_e <= 0:
            continue
        scar = (ar0 + ar1) / (s_e * math.sqrt(2))
        vol20 = float(np.nanstd(r.iloc[p - 20:p].to_numpy()))
        out.append({**e.to_dict(), "scar": float(scar), "abs_scar": abs(float(scar)), "y": int(abs(scar) > 2),
                    "vol_ratio": vol20 / s_e if s_e else 1.0, "vix": float(vix.iloc[p - 1]) if p - 1 >= 0 else np.nan,
                    "ar_bps": float((ar0 + ar1) * 1e4)})
    return pd.DataFrame(out)


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CLASSES:
        X[f"cls_{c}"] = (df["cls"] == c).astype(float)
    X["n_items"] = np.log1p(df["n_items"].astype(float))
    X["multi_filing"] = (df["filings"] > 1).astype(float)
    for t in ("pre", "intraday", "after"):
        X[f"timing_{t}"] = (df["timing"] == t).astype(float)
    X["vol_ratio"] = np.log(df["vol_ratio"].clip(0.2, 5.0))
    X["vix"] = df["vix"].fillna(df["vix"].median()).clip(9, 80) / 20.0
    return X


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def _spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def fit(events: pd.DataFrame) -> dict:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    def split(lo, hi):
        m = (events["t0"] >= lo) & (events["t0"] <= hi)
        return events[m]

    tr, va, te, po = split(*TRAIN), split(*VALID), split(*TEST), split(*POST)
    if len(tr) < 200 or tr["y"].nunique() < 2:
        raise ValueError(f"Too few training events ({len(tr)})")
    model = LogisticRegression(C=0.5, max_iter=2000)
    model.fit(features(tr), tr["y"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    z_va = model.decision_function(features(va))
    iso.fit(z_va, va["y"])

    def evaluate(df, name):
        if df.empty:
            return {"set": name, "n": 0}
        z = model.decision_function(features(df))
        p = iso.predict(z)
        impact = np.clip(np.ceil(10 * p - 1e-9), 1, 10)
        dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False) if len(df) >= 50 else None
        top = bottom = float("nan")
        if dec is not None:
            top = float(df["abs_scar"].to_numpy()[dec.to_numpy() == 9].mean())
            bottom = float(df["abs_scar"].to_numpy()[dec.to_numpy() == 0].mean())
        base = float(tr["y"].mean())
        return {
            "set": name, "n": int(len(df)), "positives": int(df["y"].sum()), "base_rate": round(float(df["y"].mean()), 4),
            "auc": round(_auc(df["y"], p), 4),
            "auc_rule_rank": round(_auc(df["y"], df["rank"] + 0.01 * df["n_items"]), 4),
            "brier": round(float(brier_score_loss(df["y"], p)), 4),
            "brier_base_rate": round(float(brier_score_loss(df["y"], np.full(len(df), base))), 4),
            "spearman_impact_abs_scar": round(_spearman(impact, df["abs_scar"]), 4),
            "top_decile_abs_scar": round(top, 3), "bottom_decile_abs_scar": round(bottom, 3),
            "reliability": reliability(df["y"].to_numpy(), p),
        }

    report = {"train": evaluate(tr, "train 2015-2020"), "valid": evaluate(va, "valid 2021"),
              "test": evaluate(te, "test 2022-2023"), "post": evaluate(po, "post-sample 2024+")}

    # Per-class base logits for the runtime model: average features, class switched on.
    Xall = features(tr)
    mean_row = Xall.mean()
    class_logit = {}
    for c in CLASSES:
        row = mean_row.copy()
        for k in CLASSES:
            row[f"cls_{k}"] = 1.0 if k == c else 0.0
        class_logit[c] = float(model.decision_function(row.to_frame().T)[0])
    by_class = (events.groupby("cls")
                .agg(n=("y", "size"), two_sigma_rate=("y", "mean"), mean_abs_scar=("abs_scar", "mean"))
                .round(4).reset_index().to_dict(orient="records"))
    xs = np.linspace(min(z_va.min(), -6), max(z_va.max(), 2), 60)
    spec = {
        "version": "event-study-8k-v1",
        "trained_on": f"{len(tr)} 8-K events {TRAIN[0][:4]}-{TRAIN[1][:4]}; isotonic on {len(va)} events in 2021",
        "class_logit": class_logit,
        "class_support": {c: int((tr["cls"] == c).sum()) for c in CLASSES},
        "default_logit": class_logit[EventClass.OTHER.value],
        "isotonic": {"x": [round(float(x), 5) for x in xs], "y": [round(float(y), 5) for y in iso.predict(xs)]},
        "diffusion_centre": 1.6,
        "coefficients": dict(zip(Xall.columns, [round(float(c), 4) for c in model.coef_[0]], strict=True)),
        "intercept": round(float(model.intercept_[0]), 4),
    }
    return {"spec": spec, "report": report, "by_class": by_class}


def reliability(y, p, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "predicted": round(float(p[m].mean()), 4),
                        "observed": round(float(y[m].mean()), 4)})
    return out


def main() -> int:
    settings = load_settings()
    root = settings.root
    market = root / "data" / "market"
    edgar = pd.read_csv(market / "edgar_8k.csv", dtype=str)
    prices = pd.read_csv(market / "prices_daily.csv", parse_dates=["date"]).set_index("date")
    events = build_events(edgar, prices)
    print(f"{len(events)} ticker-day 8-K events with SCAR labels; two-sigma share {events['y'].mean():.1%}")
    out = fit(events)
    models = root / "models"
    models.mkdir(exist_ok=True)
    (models / "impact_calibrated.json").write_text(json.dumps(out["spec"], indent=2), encoding="utf-8")
    results = root / "docs" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "impact_report.json").write_text(json.dumps({"report": out["report"], "by_class": out["by_class"]}, indent=2), encoding="utf-8")
    events.assign(t0=events["t0"].dt.strftime("%Y-%m-%d")).drop(columns=["pos"]).to_csv(
        root / "data" / "market" / "impact_events.csv", index=False, lineterminator="\n")
    for k in ("train", "valid", "test", "post"):
        r = out["report"][k]
        if r.get("n"):
            print(f"  {r['set']:20s} n={r['n']:5d} base={r['base_rate']:.3f} AUC={r['auc']:.3f} "
                  f"(rule rank {r['auc_rule_rank']:.3f}) Brier={r['brier']:.4f} (base {r['brier_base_rate']:.4f}) "
                  f"top/bottom decile |SCAR| {r['top_decile_abs_scar']:.2f}/{r['bottom_decile_abs_scar']:.2f}")
    print("Wrote models/impact_calibrated.json and docs/results/impact_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
