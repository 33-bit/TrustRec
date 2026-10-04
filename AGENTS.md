# Repository Guidelines

## Project structure

TrustRec uses Python code under `src/trustrec/`. Each subfolder has one purpose:

- `data/` reads sources and builds snapshots.
- `nlp/` creates aspect and sentiment labels.
- `graph/` builds graphs and PageRank scores.
- `recommenders/` ranks items.
- `explanations/` selects evidence and writes claims.
- `evaluation/` measures models and checks leakage.
- `schemas/` defines shared data contracts.

The Streamlit app lives in `app/`. Reproducible commands live in `scripts/`. Configuration files live in `configs/`. Tests live in `tests/`. Keep notebooks for exploration only. Keep raw data and generated model files outside Git. Store decisions, specifications, plans, and task records in `docs/adr/`, `docs/specs/`, `docs/plans/`, and `docs/tasks/`.

## Commands

Run these commands from the repository root:

```bash
make install      # install runtime and development dependencies
make format       # format Python files
make lint         # run Ruff checks
make test         # run the full pytest suite
make check        # format check, lint, and tests
make demo         # start the Streamlit application
```

Run one test file with `pytest tests/unit/test_<module>.py -q`. Record the dataset hash, snapshot ID, seed, and model hash for each experiment.

## Code style

Use Python 3.11 or newer. Use four spaces for indentation. Add type hints to public functions. Use Ruff for format and lint. Use `snake_case` for functions, variables, and files. Use `PascalCase` for classes. Use `UPPER_SNAKE_CASE` for constants. Name time fields clearly, such as `cutoff_timestamp` and `snapshot_id`.

## Tests

Name test files `test_*.py`. Name test functions `test_<behavior>`. Add contract tests for schemas. Add leakage tests for every feature pipeline. Use fixed seeds. Unit tests must not download production data.

## Commits and pull requests

Use Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `test:`, and `chore:`. Use an imperative subject. A pull request must describe the change, list validation commands and results, link its task and spec, and include screenshots for UI changes. Do not commit secrets, raw reviewer identities, or generated data.

## Security and configuration

Keep secrets and local paths in untracked environment files. Validate input paths and snapshot IDs at pipeline boundaries. Use internal IDs in the demo. Show only the evidence needed for a recommendation.
