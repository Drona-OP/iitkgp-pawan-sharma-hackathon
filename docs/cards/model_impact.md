# Model card: impact score

**Intended use.** "How big if true": the chance that an event moves its asset by more than two standard deviations, mapped to 1-10 (impact = ceil(10p)). Credibility is deliberately not an input; the trigger gate handles veracity.
**Training data.** SEC 8-K filings for the 20 universe names (data.sec.gov submissions API) joined with Yahoo Finance prices. Market model on days -250..-30, SCAR over days 0 and +1, label |SCAR| > 2.
**Model.** Logistic regression on event class, item count, timing, trailing volatility ratio and VIX; isotonic calibration on 2021; tested on 2022-2023; post-sample check 2024+. Classes the 8-K study cannot see (macro, geopolitical, product news) keep the transparent logit prior, flagged in `model_versions`. Sentiment strength and diffusion (publishers, social reach) enter as a centred prior.
**Evaluation.** AUC, Brier vs the base rate, reliability table, Spearman(impact, |SCAR|), top vs bottom decile |SCAR|, and the hand-set 8-K severity ranking as the naive baseline: `docs/results/impact_report.json`.
**Limitations.** 8-K events are a subset of news; survivorship (today's index members); acceptance times are Eastern and assumed exact.
