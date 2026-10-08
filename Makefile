PY ?= python

.PHONY: install install-ml demo replay live record api ui test lint schema

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .

install-ml:
	$(PY) -m pip install -r requirements-ml.txt

demo:
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

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

schema:
	$(PY) -m seismo schema
