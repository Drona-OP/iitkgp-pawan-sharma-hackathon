# Replay packs

A replay pack is a JSONL file of normalized `Document` records, one per line, ordered by
`published_at`. Lines starting with `#` are comments. `python -m seismo replay --pack <file>`
re-emits a pack through the full pipeline on an accelerated clock.

| Pack | Content | Status |
| --- | --- | --- |
| `demo_synthetic.jsonl` | 35 invented documents for smoke tests and the first demo. Every record is flagged `"synthetic": true`; none is real news. | Day 1 |
| `live_*.jsonl` | Recorded from live GDELT, EDGAR and Bluesky with `make record`. | Day 1, on your machine |
| DeepSeek, SVB, tariff shock, red team, quiet day | Historical windows rebuilt from GDELT and EDGAR. | Day 3 |
