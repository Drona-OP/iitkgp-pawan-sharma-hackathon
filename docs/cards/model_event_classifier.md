# Model card: event classifier

**Intended use.** Assign one of ten event classes (Macroeconomic, Geopolitical, Credit Event, M&A and Corporate Action, Product and Strategy, Earnings and Guidance, Legal and Regulatory, Operational and ESG, Management and Governance, Other) with a subtype and confidence.
**How.** SEC 8-K item codes are authoritative labels (1.03 bankruptcy, 1.05 cyber incident, 2.02 results, 4.02 restatement, 5.02 officer change). Other text uses weighted, title-boosted regular-expression rules. At event level the class is a credibility-weighted vote across the story's reports; commentary (Other) neither wins nor dilutes.
**Evaluation.** Gold-set accuracy vs the majority class in `make results` (in-sample for the rules: a sanity check, not a benchmark).
**Limitations.** Rules miss novel phrasing. Planned upgrade: an embedding classifier trained on 8-K weak labels and LLM-labelled headlines, with the LLM limited to the uncertain tail.
