PYTHON ?= python3

.PHONY: install install-laya format lint test check demo validate leakage label-laya

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-laya:
	$(PYTHON) -m pip install -e ".[laya]"

format:
	ruff format .

lint:
	ruff check .

test:
	$(PYTHON) -m pytest -q

check:
	ruff format --check .
	ruff check .
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_repo.py

leakage:
	@test -n "$(SNAPSHOT_MANIFEST)" || (echo "Set SNAPSHOT_MANIFEST=..." && exit 2)
	PYTHONPATH=src $(PYTHON) scripts/check_snapshot_leakage.py "$(SNAPSHOT_MANIFEST)"

label-laya:
	PYTHONPATH=src $(PYTHON) scripts/run_laya_labeling.py $(LAYA_ARGS)

demo:
	streamlit run app/streamlit_app.py
