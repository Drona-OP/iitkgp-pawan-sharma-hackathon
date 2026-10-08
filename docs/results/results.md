# Seismo results (generated 2026-10-08 12:41 UTC)

Sentiment backend `lexicon-v2`, impact model `logit-prior-v1`. Regenerate with `make results`.

## Replay scorecard: gated vs naive trigger

| Pack | Should trigger | Naive stress tests | Seismo auto-triggers | Held for review | Retractions | First trigger vs reference | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | yes | 4 | 1 | 0 | 0 | +25.4 h (regulators close SVB) | yes |
| deepseek 2025 | yes | 4 | 1 | 0 | 0 | +3.2 h (Monday open) | yes |
| tariff 2025 | yes | 3 | 2 | 0 | 0 | +15.2 h (first open after the announcement) | yes |
| red team | no | 1 | 0 | 1 | 1 | - | yes |
| quiet day | no | 1 | 0 | 0 | 0 | - | yes |

## Dedup and corroboration

| Pack | Documents | News/filing reports | Social posts | Stories (events) | Max independent publishers | Coordinated groups |
| --- | --- | --- | --- | --- | --- | --- |
| svb 2023 | 27 | 18 | 9 | 12 | 8 | 0 |
| deepseek 2025 | 21 | 12 | 9 | 8 | 6 | 0 |
| tariff 2025 | 18 | 13 | 5 | 9 | 5 | 0 |
| red team | 45 | 3 | 42 | 1 | 0 | 1 |
| quiet day | 13 | 10 | 3 | 11 | 1 | 0 |

## Gold set

| Component | Metric | Naive baseline | Seismo |
| --- | --- | --- | --- |
| Entity linking | precision / recall | 0.83 / 0.95 (exact alias match) | 1.00 / 0.95 (alias + context + cashtags) |
| Target sentiment | sign accuracy on 45 entity pairs | 0.78 (whole-document score) | 0.84 (entity-window score) |
| Event class | accuracy on 50 headlines | 0.24 (majority class) | 0.88 (8-K items + weighted rules) |

50 author-labelled illustrative headlines in data/gold/headlines.jsonl, incl. ambiguous names and two-company headlines with opposite sentiment. Small and in-sample for the rules; read as a sanity check.

## Latency (CPU)

| documents | engine_p50_ms | engine_p95_ms | docs_per_second | llm_calls |
| --- | --- | --- | --- | --- |
| 124 | 0.95 | 1.82 | 940.2 | 0 |

## Module A (DeepSeek replay)

NVDA first cut at 2025-01-26T16:22:00+00:00 (+22.1 h vs the Monday open); weight before the open 2.20% vs benchmark 5.00%. Turnover 10.5% vs naive tilt 59.7%.

## Module B (first stress run per pack)

| Pack | Analog | Impact | Obligor | CET1 before | CET1 after | ECL | Breaches 8% at |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svb_2023 | svb_2023 | 9 | SIVB | 13.00% | 12.90% | $73mn -> $88mn | 3.12x |
| deepseek_2025 | deepseek_2025 | 8 | NVDA | 13.00% | 13.00% | $73mn -> $73mn | > 10x |
| tariff_2025 | tariff_2025 | 8 | - | 13.00% | 11.68% | $73mn -> $170mn | 2.27x |
