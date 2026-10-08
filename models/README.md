# Models

No weights are committed. Sentiment uses `ProsusAI/finbert` from the Hugging Face Hub when
`requirements-ml.txt` is installed, and a transparent finance lexicon otherwise (the default, so
reviewers need no downloads). `impact_calibrated.json` is a few kilobytes of coefficients and an
isotonic map written by `make impact`; when it is absent the engine uses the documented logit prior.
Model cards are in `docs/cards/`.
