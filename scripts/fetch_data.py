"""Download the public market data Seismo calibrates on. Run once on a machine with internet.

    pip install yfinance pandas
    export SEISMO_EDGAR_USER_AGENT="Your Name your.email@example.com"   # SEC requires this
    python scripts/fetch_data.py                 # everything, about 2-5 minutes
    python scripts/fetch_data.py --only prices   # or: fred, edgar, caps, india, gdelt

Writes small CSVs to data/market/ (a few MB in total, safe to commit):

    prices_open.csv    adjusted opens since 2019 (for trade-at-next-open replays)
    prices_daily.csv   adjusted closes: 20 universe names, SPY, 11 sector ETFs, KRE, credit ETFs,
                       VIX, Brent, gold, dollar index, USD/INR, EUR/USD (Yahoo Finance via yfinance)
    fred_daily.csv     Treasury curve (3m, 2y, 5y, 10y, 30y), Moody's Baa/Aaa - 10y spreads,
                       ICE BofA IG / BBB / HY OAS where FRED history allows (FRED, keyless CSV)
    edgar_8k.csv       every 8-K the 20 universe companies filed, with item codes and the
                       SEC acceptance timestamp (data.sec.gov submissions API)
    caps.csv           current market capitalisation per universe ticker (SEC shares x close)
    shares_in.csv      shares outstanding for the 16 NSE names in data/universe_in.csv (yfinance)
    ../gdelt/*.csv     real article URLs, publishers and GDELT timestamps for three news windows
                       (SVB week, the Adani-Hindenburg week, a quiet control week), from the
                       GDELT 2.0 event exports (about 200-300 MB downloaded, a few MB kept)

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
UNIVERSE_IN = ROOT / "data" / "universe_in.csv"
GDELT_OUT = ROOT / "data" / "gdelt"
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
    "^NSEI": "NIFTY50",
    "^NSEBANK": "NIFTYBANK",
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
    "INDIRLTLT01STM": "IN_10Y",      # India 10-year government bond yield (OECD, monthly)
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2007-01-01"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/{name}"


def universe_rows(path: Path = UNIVERSE) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _get(url: str, headers: dict[str, str] | None = None, tries: int = 5, timeout: int = 90,
         browser: bool = False) -> bytes:
    """GET with retries. browser=True uses curl_cffi (installed with yfinance) to present a real
    Chrome TLS fingerprint, which sites behind bot protection such as FRED accept."""
    last: Exception | None = None
    if browser:
        try:
            from curl_cffi import requests as creq

            for attempt in range(tries):
                try:
                    r = creq.get(url, impersonate="chrome", timeout=timeout)
                    if r.status_code == 200 and r.content:
                        return r.content
                    last = RuntimeError(f"HTTP {r.status_code}")
                except Exception as exc:  # noqa: BLE001
                    last = exc
                time.sleep(3 * (attempt + 1))
        except ImportError:
            pass
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed public URLs
                return resp.read()
        except (OSError, TimeoutError) as exc:  # URLError, ConnectionResetError (WinError 10054), timeouts
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


# Stooq (stooq.com) is a free fallback when Yahoo is blocked or throttled.
STOOQ = {"BZ=F": "cb.f", "GC=F": "gc.f", "DX-Y.NYB": "dx.f", "INR=X": "usdinr", "EURUSD=X": "eurusd"}


def _stooq(symbol: str):
    import pandas as pd

    code = STOOQ.get(symbol, symbol.lower().replace(".", "-") + ".us")
    if symbol.startswith("^"):
        return None
    raw = _get(f"https://stooq.com/q/d/l/?s={code}&i=d", tries=3, timeout=60).decode("utf-8", "replace")
    if not raw.startswith("Date"):
        return None
    df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[df.index >= START]
    return df if len(df) else None


def _yahoo(symbol: str):
    import pandas as pd
    import yfinance as yf

    for attempt in range(3):
        try:
            df = yf.download(symbol, start=START, auto_adjust=True, progress=False, threads=False, timeout=60)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) and df["Close"].notna().any():
                return df
        except Exception as exc:  # noqa: BLE001
            print(f"[prices] {symbol}: yahoo attempt {attempt + 1} failed ({exc.__class__.__name__})")
        time.sleep(4 * (attempt + 1))
    return None


def _read_existing(name: str):
    import pandas as pd

    path = OUT / name
    if path.exists() and path.stat().st_size > 200:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return pd.DataFrame()


def fetch_prices() -> None:
    """Symbol by symbol, Yahoo first then Stooq; reruns only fetch what is still missing."""
    import pandas as pd

    try:
        import yfinance as yf

        cache = Path(os.environ.get("TEMP", "/tmp")) / "seismo-yf-cache"
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))  # avoids "database is locked" inside synced folders
    except Exception:  # noqa: BLE001
        pass
    tickers = ([r["ticker"] for r in universe_rows()] + [r["ticker"] for r in universe_rows(UNIVERSE_IN)]
               + MARKET + SECTOR_ETFS + list(YAHOO_FACTORS))
    closes, opens = _read_existing("prices_daily.csv"), _read_existing("prices_open.csv")
    todo = [t for t in tickers if YAHOO_FACTORS.get(t, t) not in closes.columns or closes[YAHOO_FACTORS.get(t, t)].isna().all()]
    print(f"[prices] {len(tickers) - len(todo)} symbols already saved, fetching {len(todo)} ...")
    got_c, got_o, failed = {}, {}, []
    for i, t in enumerate(todo, 1):
        name = YAHOO_FACTORS.get(t, t)
        df, src = _yahoo(t), "yahoo"
        if df is None:
            try:
                df, src = _stooq(t), "stooq"
            except RuntimeError:
                df = None
        if df is None:
            failed.append(t)
            print(f"[prices] {i:2d}/{len(todo)} {t:9s} FAILED")
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        got_c[name] = df["Close"]
        if "Open" in df.columns:
            got_o[name] = df["Open"]
        print(f"[prices] {i:2d}/{len(todo)} {t:9s} {len(df):5d} days from {src}")
        time.sleep(0.5)
    if got_c:
        closes = pd.concat([closes, pd.DataFrame(got_c)], axis=1).sort_index()
        closes = closes.loc[:, ~closes.columns.duplicated(keep="last")]
        closes.index.name = "date"
        closes.round(6).to_csv(OUT / "prices_daily.csv")
    if got_o:
        new_o = pd.DataFrame(got_o).loc["2019-01-01":]
        opens = pd.concat([opens, new_o], axis=1).sort_index()
        opens = opens.loc[:, ~opens.columns.duplicated(keep="last")]
        opens.index.name = "date"
        opens.round(6).to_csv(OUT / "prices_open.csv")
    print(f"[prices] saved {closes.shape[1] if len(closes) else 0} symbols; still missing: {failed or 'none'}")
    if failed:
        raise RuntimeError(f"missing {len(failed)} symbols; rerun later or on another network")


def _sec_shares(cik: str, ua: str) -> float | None:
    """Shares outstanding from SEC company facts (dei:EntityCommonStockSharesOutstanding)."""
    try:
        data = json.loads(_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json",
                               headers={"User-Agent": ua, "Accept-Encoding": "identity"}, tries=2))
        units = data["units"]["shares"]
        return float(sorted(units, key=lambda u: u.get("end", ""))[-1]["val"])
    except Exception:  # noqa: BLE001
        return None


def fetch_caps() -> None:
    """Market cap = SEC shares outstanding x last close (falls back to yfinance)."""
    ua = os.environ.get("SEISMO_EDGAR_USER_AGENT", "")
    closes = _read_existing("prices_daily.csv")
    rows = []
    for r in universe_rows():
        t, cik = r["ticker"], r["cik"].zfill(10)
        cap = float("nan")
        shares = _sec_shares(cik, ua) if ua and "example.com" not in ua else None
        if shares and t in closes.columns and closes[t].notna().any():
            cap = shares * float(closes[t].dropna().iloc[-1])
            src = "SEC shares x close"
        else:
            src = "yfinance"
            try:
                import yfinance as yf

                cap = float(yf.Ticker(t).fast_info["marketCap"])
            except Exception as exc:  # noqa: BLE001
                src = f"failed ({exc.__class__.__name__})"
        print(f"[caps] {t:5s} {cap / 1e9:10,.1f} bn  ({src})")
        rows.append({"ticker": t, "market_cap_usd": cap})
        time.sleep(0.2)
    if all(r["market_cap_usd"] != r["market_cap_usd"] for r in rows):
        raise RuntimeError("no market caps; fetch prices first")
    with (OUT / "caps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "market_cap_usd"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[caps] wrote {len(rows)} rows")


def fetch_fred() -> None:
    import pandas as pd

    existing = _read_existing("fred_daily.csv")
    frames = [existing[c] for c in existing.columns if existing[c].notna().any()] if len(existing) else []
    have = {f.name for f in frames}
    failed = []
    for sid, name in FRED_SERIES.items():
        if name in have:
            print(f"[fred] {sid:14s} already saved")
            continue
        try:
            raw = _get(FRED_URL.format(sid=sid), tries=3, browser=True).decode("utf-8")
        except RuntimeError as exc:
            print(f"[fred] {sid}: {exc}")
            failed.append(sid)
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
    if failed:
        raise RuntimeError(f"missing FRED series {failed}; rerun `--only fred` later or on another network")


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
    if (OUT / "edgar_8k.csv").exists() and (OUT / "edgar_8k.csv").stat().st_size > 1000 and not os.environ.get("SEISMO_REFETCH"):
        print("[edgar] edgar_8k.csv already saved (set SEISMO_REFETCH=1 to download again)")
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


def fetch_india() -> None:
    """Shares outstanding for the NSE names (yfinance); the replay multiplies by the close on the day."""
    rows = []
    for r in universe_rows(UNIVERSE_IN):
        t = r["ticker"]
        shares, src = float("nan"), "yfinance"
        try:
            import yfinance as yf

            tk = yf.Ticker(t)
            try:
                shares = float(tk.fast_info["shares"])
            except Exception:  # noqa: BLE001
                shares = float(tk.info.get("sharesOutstanding") or "nan")
        except Exception as exc:  # noqa: BLE001
            src = f"failed ({exc.__class__.__name__})"
        print(f"[india] {t:14s} {shares / 1e9:8.3f} bn shares  ({src})")
        rows.append({"ticker": t, "shares": shares})
        time.sleep(0.5)
    if all(r["shares"] != r["shares"] for r in rows):
        raise RuntimeError("no share counts; check the internet connection and rerun `--only india`")
    with (OUT / "shares_in.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "shares"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[india] wrote {len(rows)} rows")


# Real-news windows from the GDELT 2.0 event exports (one zip every 15 minutes since 2015).
# Each row of an export carries the first article that reported an event: its URL, and the
# 15-minute slot when GDELT saw it. We keep only articles whose URL matches the window's terms.
GDELT_WINDOWS = {
    "svb_2023": ("2023-03-08 12:00", "2023-03-10 18:00",
                 r"svb|silicon-?valley-?bank|signature-?bank|first-?republic|bank-?run|fdic|"
                 r"bank-?(?:collapse|failure|crisis)|regional-?banks?|deposit"),
    "adani_2023": ("2023-01-24 00:00", "2023-01-28 00:00", r"adani|hindenburg"),
    "control_2024": ("2024-05-07 00:00", "2024-05-09 00:00", None),   # None -> universe names
}
GDELT_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"


def _universe_terms() -> str:
    import re

    terms = set()
    for r in universe_rows():
        for alias in (r.get("aliases") or "").split("|"):
            alias = alias.strip().lower()
            if len(alias) >= 5 and alias.replace(" ", "").isalpha():
                terms.add(re.escape(alias).replace(r"\ ", "-?"))
    return "|".join(sorted(terms))


def _gdelt_slot(stamp: str, pattern) -> tuple[list[tuple[str, str, str]], bool]:
    import zipfile

    try:
        raw = _get(GDELT_URL.format(stamp=stamp), tries=3, timeout=60)
    except RuntimeError:
        return [], False
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            for line in zf.read(name).decode("utf-8", "replace").splitlines():
                cols = line.split("\t")
                if len(cols) < 61:
                    continue
                url = cols[60].strip()
                if url and pattern.search(url.lower()):
                    out.append((cols[59].strip(), url, cols[34].strip()))   # DATEADDED, SOURCEURL, AvgTone
    return out, True


def fetch_gdelt() -> None:
    import re
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta

    GDELT_OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, (start, end, terms) in GDELT_WINDOWS.items():
        path = GDELT_OUT / f"{name}.csv"
        if path.exists() and path.stat().st_size > 1000 and not os.environ.get("SEISMO_REFETCH"):
            print(f"[gdelt] {name}: already saved")
            continue
        pattern = re.compile(terms or _universe_terms())
        t0, t1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
        stamps = []
        while t0 < t1:
            stamps.append(t0.strftime("%Y%m%d%H%M%S"))
            t0 += timedelta(minutes=15)
        rows: dict[str, tuple[str, str]] = {}
        missing = 0
        print(f"[gdelt] {name}: {len(stamps)} files of about 0.3 MB ...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, (found, ok) in enumerate(pool.map(_gdelt_slot, stamps, [pattern] * len(stamps)), 1):
                missing += not ok
                for added, url, tone in found:
                    if url not in rows or added < rows[url][0]:
                        rows[url] = (added, tone)
                if i % 48 == 0 or i == len(stamps):
                    print(f"[gdelt] {name}: {i}/{len(stamps)} files, {len(rows)} matching articles")
        if missing > len(stamps) // 4:
            failed.append(name)
            print(f"[gdelt] {name}: {missing} files could not be downloaded; not saved")
            continue
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["date_added", "url", "gdelt_tone"])
            for url, (added, tone) in sorted(rows.items(), key=lambda kv: (kv[1][0], kv[0])):
                w.writerow([added, url, tone])
        print(f"[gdelt] {name}: wrote {len(rows)} articles ({missing} files missing)")
    if failed:
        raise RuntimeError(f"GDELT windows incomplete: {failed}; rerun `--only gdelt`")


STEPS = {"prices": fetch_prices, "fred": fetch_fred, "edgar": fetch_edgar, "caps": fetch_caps,
         "india": fetch_india, "gdelt": fetch_gdelt}


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
    print(f"Files are in {OUT} and {GDELT_OUT}. Zip the data folder's market and gdelt folders and upload them.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
