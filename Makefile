PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help install run test train clean

help:
	@echo "Available commands:"
	@echo "  make install   - install dependencies"
	@echo "  make run       - start the Flask app"
	@echo "  make test      - run unit tests"
	@echo "  make train     - run preprocessing + training pipeline"
	@echo "  make clean     - remove generated artifacts"

install:
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@$(PIP) install --upgrade pip
	@$(PIP) install -r src/requirements.txt

run:
	@$(PY) src/app.py

test:
	@$(PY) -m pytest -q tests

train:
	@cd src && $(PY) preprocessing.py
	@cd src && $(PY) isolation_forest.py
	@cd src && $(PY) lstm_model.py

clean:
	@rm -rf __pycache__ .pytest_cache .venv data/processed models results logs uploads
