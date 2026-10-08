"""Write data/MANIFEST.yaml: every data file with source, licence, synthetic flag, rows and SHA-256.

    python -m seismo.eval.manifest
"""

from __future__ import annotations

import hashlib

import yaml

from seismo.config import load_settings

META = {
    "universe.csv": ("SEC company_tickers.json (CIKs); aliases and domains by the author", "Public / MIT", False),
    "universe_in.csv": ("16 Nifty 50 names; NSE symbols; aliases and domains by the author", "Public / MIT", False),
    "watchlist.csv": ("Author; Harbor National Bank is fictional", "MIT", True),
    "replay/svb_2023.jsonl": ("Author: paraphrased reconstruction of public reporting", "MIT", True),
    "replay/deepseek_2025.jsonl": ("Author: paraphrased reconstruction of public reporting", "MIT", True),
    "replay/tariff_2025.jsonl": ("Author: paraphrased reconstruction of public reporting", "MIT", True),
    "replay/adani_2023.jsonl": ("Author: paraphrased reconstruction of public reporting", "MIT", True),
    "replay/svb_2023_gdelt.jsonl": ("GDELT 2.0 event exports: real URLs, publishers, timestamps; headline from URL slug", "GDELT: free, cite GDELT", False),
    "replay/adani_2023_gdelt.jsonl": ("GDELT 2.0 event exports: real URLs, publishers, timestamps; headline from URL slug", "GDELT: free, cite GDELT", False),
    "replay/control_2024_gdelt.jsonl": ("GDELT 2.0 event exports: real URLs, publishers, timestamps; headline from URL slug", "GDELT: free, cite GDELT", False),
    "gdelt/svb_2023.csv": ("GDELT 2.0 event exports (scripts/fetch_data.py --only gdelt)", "GDELT: free, cite GDELT", False),
    "gdelt/adani_2023.csv": ("GDELT 2.0 event exports (scripts/fetch_data.py --only gdelt)", "GDELT: free, cite GDELT", False),
    "gdelt/control_2024.csv": ("GDELT 2.0 event exports (scripts/fetch_data.py --only gdelt)", "GDELT: free, cite GDELT", False),
    "replay/red_team.jsonl": ("Author: fictional red-team scenario", "MIT", True),
    "replay/quiet_day.jsonl": ("Author: fictional control session", "MIT", True),
    "replay/demo_synthetic.jsonl": ("Author: fictional smoke-test pack", "MIT", True),
    "gold/headlines.jsonl": ("Author-labelled illustrative headlines", "MIT", True),
    "blotter/trade_blotter.csv": ("python -m seismo blotter (seed 2026)", "MIT", True),
    "blotter/obligors.csv": ("python -m seismo blotter (seed 2026); internal ratings synthetic", "MIT", True),
    "market/prices_daily.csv": ("Yahoo Finance via yfinance (scripts/fetch_data.py)", "Yahoo terms; research use", False),
    "market/prices_open.csv": ("Yahoo Finance via yfinance (scripts/fetch_data.py)", "Yahoo terms; research use", False),
    "market/caps.csv": ("SEC shares outstanding x last Yahoo close (scripts/fetch_data.py)", "Public / Yahoo terms; research use", False),
    "market/shares_in.csv": ("Yahoo Finance via yfinance: NSE shares outstanding (scripts/fetch_data.py)", "Yahoo terms; research use", False),
    "market/fred_daily.csv": ("FRED, Federal Reserve Bank of St. Louis (scripts/fetch_data.py)", "Public; cite FRED", False),
    "market/edgar_8k.csv": ("SEC EDGAR submissions API (scripts/fetch_data.py)", "US government public data", False),
    "market/analog_shocks.csv": ("Computed by `make shocks` from prices and FRED", "MIT", False),
    "market/impact_events.csv": ("Computed by `make impact` (8-K event study)", "MIT", False),
}


SENTFIN = ("SEntFiN 1.0 (Sinha et al., 2023, JASIST), Kaggle: ankurzing/aspect-based-sentiment-analysis-for-financial-news",
           "As published on Kaggle; research use with citation", False)


def main() -> int:
    root = load_settings().root / "data"
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".csv", ".jsonl"}):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("cache/") or rel.endswith(".db"):
            continue
        if rel.startswith("sentfin/"):
            source, licence, synthetic = SENTFIN
        else:
            source, licence, synthetic = META.get(rel, ("see data/README", "MIT", True))
        raw = path.read_bytes()
        rows = sum(1 for line in raw.decode("utf-8", "replace").splitlines() if line.strip() and not line.startswith("#"))
        entries.append({"path": f"data/{rel}", "source": source, "licence": licence, "synthetic": synthetic,
                        "rows": rows - (1 if path.suffix == ".csv" else 0), "sha256": hashlib.sha256(raw).hexdigest()})
    doc = {"note": "Every dataset Seismo uses. Nothing here is proprietary or client data. Regenerate: python -m seismo.eval.manifest",
           "datasets": entries}
    (root / "MANIFEST.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, width=120), encoding="utf-8")
    print(f"Wrote data/MANIFEST.yaml with {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
