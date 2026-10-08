"""Check data/universe.csv tickers and CIKs against SEC's public company_tickers.json.

Needs network access and a descriptive User-Agent, as SEC requires:
    SEISMO_EDGAR_USER_AGENT="Your Name your@email" python scripts/refresh_universe.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

URL = "https://www.sec.gov/files/company_tickers.json"
UNIVERSE = Path(__file__).resolve().parents[1] / "data" / "universe.csv"


def main() -> int:
    user_agent = os.environ.get("SEISMO_EDGAR_USER_AGENT", "")
    if not user_agent or "example.com" in user_agent:
        print("Set SEISMO_EDGAR_USER_AGENT to 'Your Name your@email' first.")
        return 2
    request = urllib.request.Request(URL, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - fixed SEC URL
        data = json.load(resp)
    sec = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}
    problems = 0
    with UNIVERSE.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ticker, cik = row["ticker"].upper(), row["cik"].zfill(10)
            if ticker not in sec:
                print(f"{ticker}: not found in SEC list")
                problems += 1
            elif sec[ticker] != cik:
                print(f"{ticker}: CIK {cik} in CSV, SEC says {sec[ticker]}")
                problems += 1
    print("Universe OK" if not problems else f"{problems} problem(s) found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
