"""Download the public market data Seismo calibrates on. Run once on a machine with internet.

    pip install yfinance pandas
    export SEISMO_EDGAR_USER_AGENT="Your Name your.email@example.com"   # SEC requires this
    python scripts/fetch_data.py                 # everything, about 2-5 minutes
    python scripts/fetch_data.py --only prices   # or: fred, edgar, caps

Writes small CSVs to data/market/ (a few MB in total, safe to commit):

    prices_open.csv    adjusted opens since 2019 (for trade-at-next-open replays)
    prices_daily.csv   adjusted closes: 20 universe names, SPY, 11 sector ETFs, KRE, credit ETFs,
                       VIX, Brent, gold, dollar index, USD/INR, EUR/USD (Yahoo Finance via yfinance)
    fred_daily.csv     Treasury curve (3m, 2y, 5y, 10y, 30y), Moody's Baa/Aaa - 10y spreads,
                       ICE BofA IG / BBB / HY OAS where FRED history allows (FRED, keyless CSV)
    edgar_8k.csv       every 8-K the 20 universe companies filed, with item codes and the
                       SEC acceptance timestamp (data.sec.gov submissions API)
    caps.csv           current market capitalisation per universe ticker (yfinance fast_info)

Every source is free and public. Licences and checksums are recorded in data/MANIFEST.yaml by
`python -m seismo.eval.manifest`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "market"
UNIVERSE = ROOT / "data" / "universe.csv"
START = "2007-01-01"

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE"]
MARKET = ["SPY", "KRE", "LQD", "HYG", "TLT", "GLD"]
# Yahoo symbols for non-equity factors. Column names in the CSV are the friendly names.
YAHOO_FACTORS = {
    "^VIX": "VIX",
    "BZ=F": "BRENT",
    "GC=F": "GOLD",
    "DX-Y.NYB": "DXY",
    "INR=X": "USDINR",
    "EURUSD=X": "EURUSD",
}
FRED_SERIES = {
    "DGS3MO": "UST_3M",
    "DGS2": "UST_2Y",
    "DGS5": "UST_5Y",
    "DGS10": "UST_10Y",
    "DGS30": "UST_30Y",
    "BAA10Y": "BAA_10Y",
    "AAA10Y": "AAA_10Y",
    "BAMLC0A0CM": "IG_OAS",
    "BAMLC0A4CBBB": "BBB_OAS",
    "BAMLH0A0HYM2": "HY_OAS",
    "VIXCLS": "VIX_FRED",
    "DCOILBRENTEU": "BRENT_FRED",
    "DEXINUS": "USDINR_FRED",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/{name}"


def universe_rows() -> list[dict[str, str]]:
    with UNIVERSE.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _get(url: str, headers: dict[str, str] | None = None, tries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed public URLs
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def fetch_prices() -> None:
    import pandas as pd
    import yfinance as yf

    tickers = [r["ticker"] for r in universe_rows()] + MARKET + SECTOR_ETFS + list(YAHOO_FACTORS)
    print(f"[prices] downloading {len(tickers)} symbols from {START} via yfinance ...")
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns=YAHOO_FACTORS)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index().dropna(how="all")
    missing = [t for t in tickers if YAHOO_FACTORS.get(t, t) not in close.columns or close[YAHOO_FACTORS.get(t, t)].isna().all()]
    if missing:
        print(f"[prices] WARNING: no data for {missing}")
    close.index.name = "date"
    close.round(6).to_csv(OUT / "prices_daily.csv")
    print(f"[prices] wrote {len(close)} rows x {close.shape[1]} columns")
    if isinstance(raw.columns, pd.MultiIndex) and "Open" in raw.columns.get_level_values(0):
        opens = raw["Open"].rename(columns=YAHOO_FACTORS)
        opens.index = pd.to_datetime(opens.index).tz_localize(None).normalize()
        opens = opens.sort_index().loc["2019-01-01":].dropna(how="all")  # opens only for replay windows
        opens.index.name = "date"
        opens.round(6).to_csv(OUT / "prices_open.csv")
        print(f"[prices] wrote opens: {len(opens)} rows")


def fetch_caps() -> None:
    import yfinance as yf

    rows = []
    for r in universe_rows():
        t = r["ticker"]
        try:
            cap = float(yf.Ticker(t).fast_info["marketCap"])
        except Exception as exc:  # noqa: BLE001
            print(f"[caps] {t}: {exc}")
            cap = float("nan")
        rows.append({"ticker": t, "market_cap_usd": cap})
        time.sleep(0.2)
    with (OUT / "caps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "market_cap_usd"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[caps] wrote {len(rows)} rows")


def fetch_fred() -> None:
    import pandas as pd

    frames = []
    for sid, name in FRED_SERIES.items():
        try:
            raw = _get(FRED_URL.format(sid=sid)).decode("utf-8")
        except RuntimeError as exc:
            print(f"[fred] {sid}: {exc}")
            continue
        df = pd.read_csv(io.StringIO(raw))
        date_col = df.columns[0]  # 'observation_date' (current) or 'DATE' (older format)
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df[date_col])
        s.name = name
        frames.append(s[s.index >= START])
        print(f"[fred] {sid:14s} -> {name:12s} {s.first_valid_index()} .. {s.last_valid_index()}")
        time.sleep(0.5)
    if not frames:
        print("[fred] nothing downloaded")
        return
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    out.round(4).to_csv(OUT / "fred_daily.csv")
    print(f"[fred] wrote {len(out)} rows x {out.shape[1]} columns")


def _sec_json(name: str, ua: str) -> dict:
    time.sleep(0.15)  # SEC fair-access limit is 10 requests per second
    return json.loads(_get(SEC_SUBMISSIONS.format(name=name), headers={"User-Agent": ua, "Accept-Encoding": "identity"}))


def _rows_from(block: dict, ticker: str, cik: str) -> list[dict[str, str]]:
    rows = []
    n = len(block.get("form", []))
    for i in range(n):
        form = block["form"][i]
        if form not in ("8-K", "8-K/A"):
            continue
        acc = block["accessionNumber"][i]
        doc = block.get("primaryDocument", [""] * n)[i]
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "filing_date": block["filingDate"][i],
            "acceptance_datetime": block.get("acceptanceDateTime", [""] * n)[i],
            "items": block.get("items", [""] * n)[i],
            "accession": acc,
            "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}",
        })
    return rows


def fetch_edgar() -> None:
    ua = os.environ.get("SEISMO_EDGAR_USER_AGENT", "")
    if not ua or "example.com" in ua:
        print("[edgar] set SEISMO_EDGAR_USER_AGENT='Your Name your@email' first (SEC requires it)")
        return
    rows: list[dict[str, str]] = []
    for r in universe_rows():
        ticker, cik = r["ticker"], r["cik"].zfill(10)
        try:
            data = _sec_json(f"CIK{cik}.json", ua)
        except RuntimeError as exc:
            print(f"[edgar] {ticker}: {exc}")
            continue
        got = _rows_from(data["filings"]["recent"], ticker, cik)
        for extra in data["filings"].get("files", []):
            try:
                got += _rows_from(_sec_json(extra["name"], ua), ticker, cik)
            except RuntimeError as exc:
                print(f"[edgar] {ticker} {extra['name']}: {exc}")
        got = [g for g in got if g["filing_date"] >= START]
        print(f"[edgar] {ticker:5s} {len(got):4d} 8-K filings")
        rows += got
    if not rows:
        return
    rows.sort(key=lambda x: (x["acceptance_datetime"] or x["filing_date"], x["ticker"]))
    with (OUT / "edgar_8k.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[edgar] wrote {len(rows)} filings")


STEPS = {"prices": fetch_prices, "caps": fetch_caps, "fred": fetch_fred, "edgar": fetch_edgar}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(STEPS), action="append", help="run only these steps")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in args.only or list(STEPS):
        try:
            STEPS[name]()
        except Exception as exc:  # noqa: BLE001 - one failing source must not stop the others
            print(f"[{name}] FAILED: {exc}")
            failed.append(name)
    print("\nDone." + (f" Failed steps: {failed}" if failed else " All steps succeeded."))
    print(f"Files are in {OUT}. Zip that folder and upload it to the chat.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
