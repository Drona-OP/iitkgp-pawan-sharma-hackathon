# Model card: entity-level sentiment

**Intended use.** Score how a piece of text reads for one named company, from -1 (negative) to +1 (positive), with a confidence (1 minus normalised entropy). "Infosys gains while TCS slides" must give Infosys a positive score and TCS a negative one.

**Model (default): `sentfin-target-lr-v1`.** A target-masked multinomial logistic regression trained on **SEntFiN 1.0** (Sinha et al., 2023): 10,709 Economic Times headlines with 14,345 entity-level labels.
- The company being scored becomes the token `TGT`, every other linked company `OTH`, so the model learns relative position ("TGT beats OTH") rather than company names.
- Features: word 1-2-grams of the masked sentence; the same words tagged "near" when within three tokens of `TGT`; the finance lexicon's three scores.
- Weights are stored as JSON (`models/sentiment_target.json`) and scored in pure Python; a check at training time confirms the JSON scorer reproduces scikit-learn to about 1e-7.
- CPU only, about 1 ms per document, no network, no GPU.

**Evaluation (held-out test, `make sentiment`).** 80/20 by headline, seed 2026; C=8.0 by grouped 5-fold CV on train (macro-F1 0.7993).

| System | Accuracy | Macro-F1 | Macro-F1, multi-entity | Macro-F1, conflicting |
| --- | --- | --- | --- | --- |
| Lexicon, whole headline (naive) | 0.610 | 0.608 | 0.562 | 0.456 |
| Lexicon, entity window (Seismo v1) | 0.584 | 0.575 | 0.498 | 0.469 |
| Trained, no target masking (ablation) | 0.783 | 0.785 | 0.719 | 0.508 |
| Trained, target-masked (shipped) | 0.821 | 0.820 | 0.792 | 0.691 |

Masking is what makes the model entity-aware: on headlines whose entities move in opposite directions it lifts macro-F1 from 0.51 (same model without masking) to 0.69. For reference, the SEntFiN paper reports about 93-94% for fine-tuned FinBERT and RoBERTa on GPU and about 85% for its best lexicon + gradient-boosting system; this model trades some accuracy for transparency and CPU speed.

**Fallbacks.** `lexicon-v2` (finance word lists, doubled weights for acute-stress words, negation, intensifiers, signed percentage moves) runs in CI and whenever the model file is absent. `ProsusAI/finbert` is available on request (`sentiment.backend: finbert`).

**Limitations.** Trained on Indian business headlines from 2011-2015, applied to US and Indian news; headlines only, so long articles are scored sentence by sentence. Sarcasm and multi-sentence reasoning are out of reach. Next: fine-tune a small transformer with the same masking.

**Monitoring.** Every signal records the backend in `model_versions`; Model Lab shows the evaluation table.
