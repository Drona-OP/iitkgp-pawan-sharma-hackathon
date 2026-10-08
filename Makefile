PY ?= python

.PHONY: install install-ml demo replay live record api ui test lint schema data shocks blotter impact packs results all-offline

install:            ## dependencies + editable install
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .

install-ml:         ## optional: real FinBERT sentiment (CPU torch)
	$(PY) -m pip install -r requirements-ml.txt

demo:               ## dashboard + API + default replay, no keys
	$(PY) -m seismo demo

replay:
	$(PY) -m seismo replay --reset

live:
	$(PY) -m seismo live

record:
	$(PY) -m seismo live --minutes 45 --record data/replay/live_$$(date +%Y%m%d_%H%M).jsonl

api:
	$(PY) -m seismo api

ui:
	$(PY) -m seismo ui

data:               ## download public market data (Yahoo, FRED, SEC) into data/market
	$(PY) scripts/fetch_data.py

shocks:             ## historical-analog shock vectors from data/market
	$(PY) -m seismo shocks

blotter:            ## synthetic trade blotter -> data/blotter
	$(PY) -m seismo blotter

impact:             ## 8-K event study -> models/impact_calibrated.json
	$(PY) -m seismo.eval.impact_study

packs:              ## rebuild the five replay packs
	$(PY) scripts/build_replay_packs.py

results:            ## regenerate every number in the deck -> docs/results/ (+ manifest)
	$(PY) -m seismo.eval.results
	$(PY) -m seismo.eval.manifest

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

schema:
	$(PY) -m seismo schema
