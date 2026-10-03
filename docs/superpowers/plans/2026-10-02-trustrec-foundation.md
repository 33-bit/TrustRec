# TrustRec Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a reproducible TrustRec repository with typed contracts, quality gates, protocol documentation, and a task backlog.

**Architecture:** Keep research code under `src/trustrec/`, expose stable contracts from `schemas`, and use `Makefile` targets as the contributor interface. External data and generated artifacts are referenced by manifests and hashes rather than committed.

**Tech Stack:** Python 3.11+, setuptools, pytest, Ruff, Parquet/DuckDB, scikit-learn, sparse graph tooling, and Streamlit.

**Spec:** `docs/superpowers/specs/2026-10-02-trustrec-foundation-design.md`

## Global Constraints

- Use global temporal cutoffs and reject future evidence.
- Keep raw reviews, generated models, and caches outside Git.
- Do not describe PageRank as a reviewer-trust label or ranking scores as purchase probabilities.
- Make unit tests deterministic and independent of downloaded production data.

---

### Task 1: Repository harness

**Files:** `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`, `CONTRIBUTING.md`

- [x] Define Python package metadata, dependency groups, pytest paths, and Ruff rules.
- [x] Add `make install`, `format`, `lint`, `test`, `check`, `validate`, and `demo` targets.
- [x] Exclude secrets, raw data, caches, and generated artifacts.
- [ ] Run `make install` in a clean virtual environment and record the environment lock decision.

### Task 2: Snapshot-aware contracts

**Files:** `src/trustrec/schemas/contracts.py`, `tests/unit/test_contracts.py`

- [x] Implement `SnapshotManifest`, `EvidenceRef`, and `RecommendationRecord` validation.
- [x] Add tests for timezone-aware provenance, confidence bounds, and positive ranks.
- [ ] Extend the contracts with explicit schema versions after the first Parquet schema is profiled.

### Task 3: Protocol and architecture records

**Files:** `docs/adr/*.md`, `docs/specs/*.md`, `configs/protocol.toml`

- [x] Record scope, temporal split, stack, reproducibility, and evidence decisions.
- [x] Define table fields, candidate/target rules, metrics, and demo behavior.
- [x] Version initial protocol defaults without claiming measured outcomes.

### Task 4: Contributor and task workflow

**Files:** `AGENTS.md`, `docs/plans/roadmap.md`, `docs/tasks/backlog.md`, `docs/tasks/task-template.md`

- [x] Document module ownership boundaries, commands, style, tests, and PR expectations.
- [x] Decompose delivery into phases with dependencies and acceptance checks.
- [ ] Assign owners and dates after the team confirms availability.

### Task 5: Verify the foundation

- [ ] Run `python scripts/validate_repo.py`.
- [ ] Run `python -m pytest -q` and confirm all contract tests pass.
- [ ] Run `ruff format --check .` and `ruff check .` after dependencies are installed.
- [ ] Review every spec for placeholders, contradictions, and unsupported result claims.

