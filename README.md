# TrustRec

TrustRec is a capstone research system for product recommendation with evidence. It uses ratings, product aspects, review opinions, and a user-item graph. It returns top-K items with source evidence.

The repository follows a data-contract-first process. A design target is not a result. A result must come from a recorded experiment.

Amazon Reviews 2023 by McAuley Lab remains the main source. The project uses an LLM for aspect, sentiment, and evidence labels. `development_pilot` data has status `llm_silver` and can refine prompts or guidelines. A separate frozen `llm_pseudo_test` pass supports consistency checks. `recommendation_test` uses real Amazon temporal interactions and ratings. LLM labels are not human gold labels, and the full corpus does not need manual labeling.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make check
make demo
```

`make install` installs the exact versions in `requirements.lock`. The tested interpreter version is in `.python-version`.

The current demo is a small shell that validates the serving contract. The repository does not include Amazon Reviews files. Put local downloads under `data/raw/`. Create a snapshot manifest and record the dataset hash before you run an experiment.

The full Video Games benchmark snapshot uses `make snapshot-full`. It scans Amazon Reviews 2023 in batches and writes generated tables under `data/processed/`. The reviewable manifest is `docs/tasks/t1_2_full_snapshot_manifest.json`.

## Repository map

- `src/trustrec/`: production modules and typed contracts.
- `scripts/`: reproducible command-line entry points.
- `configs/`: versioned protocol and model configuration.
- `tests/`: contract, unit, and integration tests.
- `app/`: Streamlit demonstration.
- `docs/adr/`: architectural decisions.
- `docs/specs/`: stable data and API contracts.
- `docs/plans/` and `docs/tasks/`: plans and tasks.
- `docs/specs/`: research, data, model, evaluation, serving, and report contracts.
- `docs/design/` and `docs/reference/`: the source design, terms, and claims policy.
- `schemas/`: machine-readable artifact and annotation schemas.

Read [AGENTS.md](AGENTS.md) before contributing. The supplied design is preserved at `docs/design/trustrec-design-source.md`. The active repository decision is [ADR-0013](docs/adr/0013-llm-only-annotation-policy.md).
