# Full Data and Evaluation Implementation Plan

> **For agentic workers:** Use the repository task workflow and verify each task before moving to the next task.

**Goal:** Replace the bounded pilot snapshot with a full Video Games category snapshot and create the held-out LLM evaluation split that the design requires.

**Architecture:** Stream Amazon Reviews 2023 with range requests. Join reviews to the full metadata scan before the software-game filter. Write normalized rows in batches, then derive time-safe Parquet tables with DuckDB. Keep the 100-unit `development_pilot` separate from the frozen `llm_pseudo_test` split.

**Tech Stack:** Python 3.11+, urllib range requests, PyArrow Parquet, DuckDB, pytest, Ruff.

**Spec:** `docs/specs/data-lineage-and-snapshot.md`, `docs/specs/nlp-annotation-contract.md`, `docs/adr/0013-llm-only-annotation-policy.md`.

## Global Constraints

- Keep Amazon Reviews 2023 by McAuley Lab as the source.
- Keep the source review text unchanged.
- Apply the metadata scope filter before model data use.
- Use global temporal cutoffs and keep `T0 < T1`.
- Keep recommendation interactions separate from LLM label evaluation.
- Never tune on a frozen `llm_pseudo_test` split.
- Keep raw source files and generated Parquet files outside Git.

### Task 1: Full category profile

- [x] Add a full sequential profile mode with source SHA-256 and complete counts.
- [x] Run the profile for Video Games and record missingness, scope counts, and retention.
- [x] Update the T1.1 memo and artifact with full-category evidence.

### Task 2: Full temporal snapshot

- [x] Add a batch streaming build path for the full filtered category.
- [x] Derive train, fit, validation, and recommendation test tables without a row cap.
- [x] Write a benchmark manifest with source, artifact, cutoff, and scope hashes.

### Task 3: Full leakage gate

- [x] Run leakage checks on every full snapshot table.
- [x] Add tests for full-mode provenance and deterministic hashes.
- [x] Update T1.3 and the backlog with the full evidence.

### Task 4: Held-out LLM evaluation

- [x] Keep T2.1 as the development pilot and attach it to the full snapshot.
- [x] Select a disjoint held-out source split for `llm_pseudo_test`.
- [x] Generate labels with recorded model and prompt provenance.
- [x] Freeze its manifest before reporting consistency results.

### Task 5: Verification

- [ ] Run `make check`, `make validate`, both validators, and `compileall`.
- [ ] Record the full snapshot and held-out split hashes in the task reports.
