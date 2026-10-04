# T0.2: Python and dependency versions

Status: Done on 2026-10-04.

This task locks the interpreter and package versions for the repository harness. The lock covers the runtime and development packages in `pyproject.toml`.

Related records are [ADR-0003](../adr/0003-python-parquet-duckdb-streamlit.md), the [foundation design](../superpowers/specs/2026-10-02-trustrec-foundation-design.md), the [foundation plan](../superpowers/plans/2026-10-02-trustrec-foundation.md), and the [task backlog](backlog.md). This task uses no source dataset and has no temporal cutoff.

## Locked files

- `.python-version` contains `3.14.4`.
- `requirements.lock` contains exact versions for the runtime and development environment.
- `Makefile` installs the lock before it installs the project in editable mode.
- `t0_2-install.log` records the clean install output.

A transitive dependency is a package that another package needs. The lock includes these packages so that a fresh environment uses the same versions.

## Reproduce the environment

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

`make install` reads `requirements.lock` and then installs the project with `--no-deps`. This keeps the package versions under the lock file.

The lock was generated with Python 3.14.4 on macOS arm64. The project still declares Python 3.11 or newer in `pyproject.toml` for source compatibility.

## Recorded validation

The clean install completed in a new virtual environment. `pip check` returned `No broken requirements found.`

The lock SHA-256 is `caa3c7a16e535ac60c34c1df6b3b11e6bb5cfbbdd309e1d8fec1efde287d2547`.

The repository checks produced these results:

- `make validate`: `Validated 20 repository foundation files`.
- `ruff format --check .`: `100 files already formatted`.
- `ruff check .`: `All checks passed!`.
- `python -m pytest -q`: `26 passed in 0.71s`.

The install output is in [`t0_2-install.log`](t0_2-install.log). The log replaces local paths with placeholders.

## Update rule

Create a new lock when the Python version or a dependency declaration changes. Record the generation date, interpreter version, and new lock hash in this task record.
