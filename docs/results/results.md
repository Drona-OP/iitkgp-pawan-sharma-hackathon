# Seismo results (generated 2026-10-08 21:26 UTC)

Sentiment backend `lexicon-v2`, impact model `event-study-8k-v1`. Regenerate with `make results`.

## Replay scorecard: gated vs naive trigger

| Pack | Should trigger | Naive stress tests | Seismo auto-triggers | Held for review | Retractions | First trigger vs reference | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | yes | 4 | 1 | 0 | 0 | 25.4 h before (regulators close SVB) | yes |
| deepseek 2025 | yes | 4 | 1 | 0 | 0 | 3.2 h before (Monday open) | yes |
| tariff 2025 | yes | 3 | 2 | 0 | 0 | 15.2 h before (first open after the announcement) | yes |
| red team | no | 1 | 0 | 1 | 1 | - | yes |
| quiet day | no | 1 | 0 | 0 | 0 | - | yes |

## Dedup and corroboration

| Pack | Documents | News/filing reports | Social posts | Stories (events) | Max independent publishers | Coordinated groups |
| --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | 27 | 18 | 9 | 12 | 8 | 0 |
| deepseek 2025 | 21 | 12 | 9 | 7 | 6 | 0 |
| tariff 2025 | 18 | 13 | 5 | 9 | 5 | 0 |
| red team | 45 | 3 | 42 | 1 | 0 | 1 |
| quiet day | 13 | 10 | 3 | 11 | 1 | 0 |

## Gold set

| Component | Metric | Naive baseline | Seismo |
| --- | --- | --- | --- |
| Entity linking | precision / recall | 0.83 / 0.95 (exact alias match) | 1.00 / 0.95 (alias + context + cashtags) |
| Target sentiment | sign accuracy on 45 entity pairs | 0.76 (whole-document score) | 0.87 (entity-window score) |
| Event class | accuracy on 50 headlines | 0.24 (majority class) | 0.88 (8-K items + weighted rules) |

50 author-labelled illustrative headlines in data/gold/headlines.jsonl, incl. ambiguous names and two-company headlines with opposite sentiment. Small and in-sample for the rules; read as a sanity check.

## Latency (CPU)

| documents | engine_p50_ms | engine_p95_ms | docs_per_second | llm_calls |
| --- | --- | --- | --- | --- |
| 124 | 0.79 | 1.63 | 1103.6 | 0 |

## Module A (DeepSeek replay)

NVDA first cut at 2025-01-26T16:22:00+00:00 (22.1 h before the Monday open); weight before the open 10.04% vs benchmark 15.00%. Turnover 19.0% vs naive tilt 180.1%.

| Portfolio | Return | Max drawdown |
| --- | --- | --- |
| Seismo | +0.78% | -2.76% |
| Benchmark | +0.64% | -3.11% |
| Naive tilt | +0.68% | -2.83% |

## Module B (most severe auto-triggered stress run per pack)

| Pack | Analog | Impact | Obligor | CET1 before | CET1 after | ECL | Breaches 8% at |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svb_2023 | svb_2023 | 10 | SIVB | 13.00% | 12.16% | $73mn -> $292mn | 3.11x |
| deepseek_2025 | deepseek_2025 | 8 | NVDA | 13.00% | 13.13% | $73mn -> $73mn | > 10x |
| tariff_2025 | tariff_2025 | 9 | - | 13.00% | 10.87% | $73mn -> $212mn | 1.87x |
