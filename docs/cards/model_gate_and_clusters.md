# Model card: event clustering and the trigger gate

**Clusters.** One entity and one event class within a rolling 24-hour chain form an event. Near-duplicates (MinHash >= 0.8) inherit the original publisher. Owner groups (e.g. WSJ, MarketWatch and Barron's) count once. Five or more accounts posting near-identical text within an hour form a coordinated group, which counts as one voice. Lookalike domains carrying a known brand are flagged and never corroborate.
**Denials.** Denial language from an SEC filing, an authoritative wire or regulator, or the issuer's own domain marks the matched story as retracted; its weight in Module A drops to zero.
**Gate.** TRIGGER only when impact >= 8, the class/subtype maps to an analog, confidence >= 0.7, and there are >= 3 independent publishers within 60 minutes or one authoritative source, and no denial. Otherwise REVIEW (human in the loop) or LOG.
**Evaluation.** Replay scorecard in `make results`: triggers on SVB, DeepSeek and the tariff shock; holds and then retracts the red-team fake; no trigger on the quiet day; compared with a naive "10 x |sentiment| > 7" trigger.
