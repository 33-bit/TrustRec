# Repository Guidelines

## Project Structure & Module Organization

TrustRec is a Python research project. Keep production code under `src/trustrec/`, split by responsibility: `data/`, `nlp/`, `graph/`, `recommenders/`, `explanations/`, `evaluation/`, and `schemas/`. Put the Streamlit demo in `app/`, reproducible command-line entry points in `scripts/`, and configuration files in `configs/`.

Use `tests/` for contract, unit, and integration tests. Keep exploratory notebooks in `notebooks/`; do not make notebooks the only implementation of a pipeline. Store only small fixtures and manifests in `data/`; raw Amazon Reviews files and generated model artifacts stay outside Git. Project decisions, specifications, plans, and task breakdowns belong in `docs/adr/`, `docs/specs/`, `docs/plans/`, and `docs/tasks/`.

## Build, Test, and Development Commands

Use the repository Make targets as the stable interface:

```bash
make install      # install runtime and development dependencies
make format       # format Python files
make lint         # run Ruff checks
make test         # run the full pytest suite
make check        # format check, lint, and tests
make demo         # start the Streamlit application
```

Run focused tests with `pytest tests/unit/test_<module>.py -q`. Record dataset, snapshot, seed, and model hashes for every experiment.

## Coding Style & Naming Conventions

Use Python 3.11+, four-space indentation, type hints on public functions, and small modules with one clear responsibility. Ruff is the formatter and linter. Use `snake_case` for functions, variables, and files; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Name snapshot-aware values explicitly, for example `cutoff_timestamp` and `snapshot_id`.

## Testing Guidelines

Use pytest. Test files are named `test_*.py`, and test functions are named `test_<behavior>`. Add contract tests for schemas and temporal leakage checks for every feature-producing pipeline. Keep tests deterministic with fixed seeds; never require a downloaded production dataset for unit tests.

## Commit & Pull Request Guidelines

Use Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `test:`, and `chore:`; keep the subject imperative and concise. Pull requests must describe the research or code change, list validation commands and results, link the relevant task/spec, and include screenshots for dashboard changes. Do not commit credentials, raw reviewer identities, or generated model/data artifacts.

## Security & Configuration Tips

Keep secrets and local paths in untracked environment files. Validate input paths and snapshot IDs at pipeline boundaries. Do not expose real reviewer identity in the demo; use internal IDs and cite only the evidence required to explain a recommendation.
