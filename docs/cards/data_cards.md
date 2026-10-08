# Data cards

| Data | Source | Licence | Synthetic | Notes |
| --- | --- | --- | --- | --- |
| `data/universe.csv` | SEC company_tickers.json for CIKs; aliases and issuer domains written by the author | Public / MIT | No | 20 S&P 100 names, 11 GICS sectors |
| `data/universe_in.csv` | NSE symbols; aliases, domains and business group written by the author | Public / MIT | No | 16 Nifty 50 names (India index; Adani group tagged) |
| `data/watchlist.csv` | Author | MIT | Partly | Regional banks for the SVB replay; "Harbor National Bank" is fictional |
| `data/replay/svb_2023`, `deepseek_2025`, `tariff_2025`, `adani_2023` | Author, paraphrasing widely reported facts | MIT | Yes | Reconstructions; `.example` publishers (RFC 2606); invented social posts |
| `data/replay/red_team`, `quiet_day`, `demo_synthetic` | Author | MIT | Yes | Fictional |
| `data/replay/*_gdelt.jsonl`, `data/gdelt/*.csv` | GDELT 2.0 event exports (15-minute files) | Free; cite GDELT | No | Real article URLs, publishers and GDELT timestamps for the SVB week, the Adani week and a quiet control week; headlines rebuilt from URL slugs |
| `data/sentfin/SEntFiN-v1.1.csv` | SEntFiN 1.0 (Sinha et al., 2023), Kaggle | As published; research use with citation | No | 10,753 Economic Times headlines, 14,404 entity-level sentiment labels; trains the sentiment model |
| `data/gold/headlines.jsonl` | Author | MIT | Yes | 50 labelled illustrative headlines |
| `data/blotter/*` | `python -m seismo blotter` (seed 2026) | MIT | Yes | Invented exposures; public names only as obligors; internal ratings are synthetic, not agency ratings |
| `data/market/prices_daily.csv`, `prices_open.csv`, `shares_in.csv` | Yahoo Finance via yfinance | Yahoo terms, research use | No | US and NSE prices, Nifty 50, Nifty Bank; NSE shares outstanding. Fetched by `make data` |
| `data/market/caps.csv` | SEC shares outstanding x last Yahoo close | Public / Yahoo terms | No | US benchmark caps |
| `data/market/fred_daily.csv` | FRED (St. Louis Fed) | Public, cite FRED | No | Treasury curve, Baa/Aaa spreads, ICE BofA OAS where available |
| `data/market/edgar_8k.csv` | SEC EDGAR submissions API | US government public data | No | 8-K item codes and acceptance times |
| `data/market/analog_shocks.csv` | Computed by `make shocks` | MIT | No | Largest move per factor inside each crisis window |
| PD table (`module_b/credit.py`) | Smoothed from S&P Global Ratings' public annual default studies | Public research, cited | No | By-notch smoothing is ours |

No S&P Global or Crisil client data, and no proprietary data, is used anywhere.
