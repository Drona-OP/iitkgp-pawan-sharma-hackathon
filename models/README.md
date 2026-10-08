# Models

Model weights are not committed. Day 1 uses `ProsusAI/finbert` straight from the Hugging Face Hub
(or the lexicon fallback when `transformers` is not installed). From Day 2 the fine-tuned,
target-masked sentiment model and the event classifier are published to the Hugging Face Hub and
downloaded on first run. Each model ships with a model card in this folder.
