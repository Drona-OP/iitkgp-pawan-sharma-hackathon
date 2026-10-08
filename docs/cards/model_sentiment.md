# Model card: entity-level sentiment

**Intended use.** Score how a piece of text reads for one named entity, from -1 (negative) to +1 (positive), with a confidence (1 minus normalised entropy).
**How.** The engine scores only the sentence (and, when two companies share a sentence, the clause) that mentions the entity, so "Microsoft gains as Google loses" yields +/- rather than one blended score. Backends: `ProsusAI/finbert` (optional) or `lexicon-v2`: finance word lists, doubled weights for acute-stress words, negation, intensifiers and signed percentage moves.
**Evaluation.** `make results`, gold set (author-labelled, 50 headlines): entity-window sign accuracy vs whole-document score. Financial PhraseBank evaluation needs Hugging Face access and is listed as next work.
**Limitations.** The lexicon misses sarcasm and target-dependent phrases ("gains traction" for a rival). Not fine-tuned in this build; the target-masked fine-tune (SEntFiN, FiQA) is the planned upgrade.
**Monitoring.** Model Lab shows the backend name on every signal (`model_versions`).
