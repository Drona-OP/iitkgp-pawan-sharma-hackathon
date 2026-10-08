# Seismo results (generated 2026-10-08 23:11 UTC)

Sentiment backend `sentfin-target-lr-v1`, impact model `event-study-8k-v1`. Regenerate with `make results`.

## Replay scorecard: gated vs naive trigger

| Pack | Should trigger | Naive stress tests | Seismo auto-triggers | Held for review | Retractions | First trigger vs reference | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | yes | 9 | 3 | 1 | 0 | 39.9 h before (regulators close SVB) | yes |
| deepseek 2025 | yes | 6 | 1 | 0 | 0 | 3.4 h before (Monday open) | yes |
| tariff 2025 | yes | 9 | 3 | 1 | 0 | 16.4 h before (first open after the announcement) | yes |
| adani 2023 | yes | 7 | 3 | 0 | 0 | 13.9 h before (first NSE open after the report) | yes |
| red team | no | 1 | 0 | 1 | 1 | - | yes |
| quiet day | no | 2 | 0 | 0 | 0 | - | yes |

## Dedup and corroboration

| Pack | Documents | News/filing reports | Social posts | Stories (events) | Max independent publishers | Coordinated groups |
| --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | 27 | 18 | 9 | 12 | 8 | 0 |
| deepseek 2025 | 21 | 12 | 9 | 7 | 6 | 0 |
| tariff 2025 | 18 | 13 | 5 | 9 | 5 | 0 |
| adani 2023 | 28 | 19 | 9 | 17 | 5 | 0 |
| red team | 45 | 3 | 42 | 1 | 0 | 1 |
| quiet day | 13 | 10 | 3 | 11 | 1 | 0 |

## Real news (GDELT): the same engine on real article URLs, publishers and timestamps

| Window | Real articles | Publishers | Stories (events) | Naive stress tests | Seismo auto-triggers | Held for review | First trigger vs reference | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | 45 | 32 | 5 | 7 | 1 | 0 | 0.2 h before (regulators close SVB) | yes |
| adani 2023 | 163 | 64 | 19 | 47 | 1 | 1 | 5.5 h after (first NSE open after the report) | yes |
| control 2024 | 816 | 495 | 59 | 48 | 0 | 2 | - | yes |

Headlines are rebuilt from URL slugs; timestamps are GDELT's 15-minute ingestion slots, which can trail publication by 15-30 minutes. First triggers: svb_2023 at 2023-03-10T16:00:00+00:00 on "SVB CEO Becker Asks Silicon Valley Bank Clients to Stay Calm"; adani_2023 at 2023-01-25T09:15:00+00:00 on "Adani Stocks Drop after Hindenberg Accuses Firm of Manipulation and Fraud".

## Target sentiment on real labelled headlines (SEntFiN 1.0, held-out test)

| System | Accuracy | Macro-F1 | Macro-F1, multi-entity headlines | Macro-F1, conflicting headlines |
| --- | --- | --- | --- | --- |
| Lexicon, whole headline (naive) | 0.610 | 0.608 | 0.562 | 0.456 |
| Lexicon, entity window (Seismo v1) | 0.584 | 0.575 | 0.498 | 0.469 |
| Trained, no target masking (ablation) | 0.783 | 0.785 | 0.719 | 0.508 |
| Trained, target-masked (shipped) | 0.821 | 0.820 | 0.792 | 0.691 |

SEntFiN 1.0 (Sinha et al., 2023): Economic Times headlines with entity-level labels; 14345 entity pairs from 10709 headlines. 80/20 by headline, seed 2026; C=8.0 by grouped 5-fold CV on train (macro-F1 0.7993). Published reference: SEntFiN paper, test set: finBERT and RoBERTa fine-tuned reach about 93-94% (GPU transformers); best lexicon + GBM about 85%.

## Gold set

| Component | Metric | Naive baseline | Seismo |
| --- | --- | --- | --- |
| Entity linking | precision / recall | 0.83 / 0.95 (exact alias match) | 1.00 / 0.95 (alias + context + cashtags) |
| Target sentiment | sign accuracy on 45 entity pairs | 0.73 (whole-document score) | 0.82 (entity-window score) |
| Event class | accuracy on 50 headlines | 0.24 (majority class) | 0.88 (8-K items + weighted rules) |

50 author-labelled illustrative headlines in data/gold/headlines.jsonl, incl. ambiguous names and two-company headlines with opposite sentiment. Small and in-sample for the rules; read as a sanity check.

## Latency (CPU)

| documents | engine_p50_ms | engine_p95_ms | docs_per_second | llm_calls |
| --- | --- | --- | --- | --- |
| 152 | 1.31 | 2.86 | 661.2 | 0 |

## Module A (DeepSeek replay)

NVDA first cut at 2025-01-26T16:22:00+00:00 (22.1 h before the Monday open); weight before the open 10.04% vs benchmark 15.00%. Turnover 36.0% vs naive tilt 192.4%.

| Portfolio | Return | Max drawdown |
| --- | --- | --- |
| Seismo | +0.16% | -2.76% |
| Benchmark | +0.18% | -3.11% |
| Naive tilt | -0.57% | -2.45% |

## Module A, India index (Adani-Hindenburg replay)

| Name | Benchmark | Weight before the 25 Jan open | First cut | Circuit breaker |
| --- | --- | --- | --- | --- |
| ADANIENT | 6.25% | 1.25% | 2023-01-24T13:00:00+00:00 (14.8 h before the open) | yes |
| ADANIPORTS | 6.25% | 6.76% | 2023-01-25T10:20:00+00:00 (6.6 h after the open) | no |

Turnover 27.0% vs naive tilt 181.2%.

## Module B (most severe auto-triggered stress run per pack)

| Pack | Analog | Impact | Obligor | CET1 before | CET1 after | ECL | Breaches 8% at |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svb_2023 | svb_2023 | 10 | SIVB | 13.00% | 12.39% | $76mn -> $294mn | 3.26x |
| deepseek_2025 | deepseek_2025 | 8 | NVDA | 13.00% | 13.12% | $76mn -> $76mn | > 10x |
| tariff_2025 | tariff_2025 | 9 | AAPL | 13.00% | 10.96% | $76mn -> $220mn | 1.89x |
| adani_2023 | adani_2023 | 8 | ADANIENT.NS | 13.00% | 12.75% | $76mn -> $83mn | > 10x |
