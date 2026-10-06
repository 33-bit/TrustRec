PYTHON ?= python3

.PHONY: install format lint test check demo report validate leakage snapshot-full

install:
	$(PYTHON) -m pip install --requirement requirements.lock
	$(PYTHON) -m pip install --no-deps --no-build-isolation -e .

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

snapshot-full:
	PYTHONPATH=src $(PYTHON) scripts/build_full_snapshot.py --category Video_Games

demo:
	streamlit run app/streamlit_app.py

report:
	PYTHONPATH=src $(PYTHON) scripts/package_report.py --spec configs/report_bundle.json --output-dir reports/generated/t5-3
