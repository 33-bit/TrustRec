# T5.2 Streamlit Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic five-minute Streamlit demo that reads prepared TrustRec records, compares a baseline with TrustRec, shows evidence and support limits, and applies temporary aspect priorities.

**Architecture:** A pure Python service in `app/demo.py` validates one versioned JSON bundle, selects prepared rankings, applies temporary aspect priorities, and returns display records. `app/streamlit_app.py` renders controls and records without fitting or downloading data. `app/demo_bundle.json` supplies a small synthetic fallback for a complete five-minute flow when external artifacts are absent.

**Tech Stack:** Python 3.11+, Streamlit, standard library JSON and dataclasses, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-05-t5-2-streamlit-demo-design.md`

## Global Constraints

- Read prepared artifacts only. Do not download raw data or fit a model during a request.
- Keep `snapshot_id`, `dataset_hash`, cutoff, configuration hash, model hashes, and run IDs with displayed results.
- Keep learned user weights separate from temporary aspect priorities.
- Show source review IDs, evidence text, and timestamps. Do not show reviewer identities.
- Use stable item ID tie-breaking and fixed synthetic fixture values.
- Keep raw data and generated model files outside Git. The checked-in bundle is a small synthetic demonstration fixture.
- Use fixed test inputs. Do not download production data in tests.

### Task 1: Define the bundle contract and write the fixture

**Files:**
- Create: `app/demo_bundle.json`
- Create: `tests/unit/test_demo.py`

**Interfaces:**
- The fixture exposes schema version `1`, one snapshot, one low-history user, three items, baseline and TrustRec rankings, evidence rows, and one measured metric.
- The tests use `Path("app/demo_bundle.json")` as the deterministic source.

- [x] **Step 1: Write the failing fixture shape test.**

  Assert that the fixture contains lineage fields, one user with history count at most two, both `b0_most_popular` and `t0_trustrec`, evidence rows, and a metric row.

- [x] **Step 2: Run the focused test and confirm the expected missing-file failure.**

  Run `python3 -m pytest tests/unit/test_demo.py -q` and record the failure before adding the fixture.

- [x] **Step 3: Add the smallest synthetic bundle.**

  Include a low-history user, three named game items, learned weights for gameplay, graphics, and story, baseline scores, TrustRec score parts, positive evidence, conflicting evidence, insufficient support, source timestamps, and a metric with a run ID.

- [x] **Step 4: Run the focused test and confirm the fixture shape passes.**

  Run `python3 -m pytest tests/unit/test_demo.py -q` and keep the fixture values fixed for later tests.

### Task 2: Implement strict bundle loading and lineage validation

**Files:**
- Create: `app/demo.py`
- Modify: `tests/unit/test_demo.py`

**Interfaces:**
- `DemoValidationError(ValueError)` reports invalid bundle records.
- `load_demo_bundle(path: Path) -> DemoBundle` reads and validates one JSON object.
- `load_demo_bundle_with_fallback(configured_path: Path | None = None) -> DemoLoadResult` returns the configured bundle when valid, otherwise the fallback bundle with `used_fallback` and `load_error` fields.
- `DemoBundle` exposes `snapshot_id`, `dataset_hash`, `cutoff_timestamp`, `configuration_hash`, `model_hashes`, `users`, `items`, `rankings`, `evidence`, and `metrics`.

- [x] **Step 1: Add tests for valid loading and invalid lineage.**

  Assert that valid data becomes typed records. Copy the fixture to a temporary file, remove `dataset_hash`, and assert that `DemoValidationError` names the missing field.

- [x] **Step 2: Run the tests and confirm they fail because the loader is absent.**

  Run `python3 -m pytest tests/unit/test_demo.py -q`.

- [x] **Step 3: Implement the typed records and field validation.**

  Validate non-empty IDs, finite scores, ISO timestamps, supported schema version, unique candidate IDs, matching ranking item IDs, and complete lineage. Ignore unknown fields so prepared artifacts can carry extra metadata.

- [x] **Step 4: Implement configured-path and fallback loading.**

  Resolve `TRUSTREC_DEMO_BUNDLE` when the caller does not pass a path. Fall back only after a missing or invalid configured file. Preserve the validation message for the view.

- [x] **Step 5: Run the focused loader tests and confirm they pass.**

  Run `python3 -m pytest tests/unit/test_demo.py -q`.

### Task 3: Implement ranking views and temporary priorities

**Files:**
- Modify: `app/demo.py`
- Modify: `tests/unit/test_demo.py`

**Interfaces:**
- `rank_recommendations(bundle: DemoBundle, *, user_id: str, model_id: str, k: int, priorities: Mapping[str, float] | None = None) -> tuple[DemoRecommendation, ...]` returns stable ranked records.
- `compare_rankings(before: Sequence[DemoRecommendation], after: Sequence[DemoRecommendation]) -> tuple[str, ...]` returns item IDs whose rank changed.
- `DemoRecommendation` exposes item title, item ID, rank, total score, score parts, raw aspect scores, support, learned weights, temporary priorities, effective weights, and evidence state.

- [x] **Step 1: Add tests for stable baseline ranking and K limiting.**

  Assert that baseline scores sort by descending score with item ID tie-breaking and that K returns only the requested number of items.

- [x] **Step 2: Add tests for priority isolation and a visible rank change.**

  Rank TrustRec with empty priorities and with a graphics priority. Assert that the learned weights stay unchanged, the priority map appears only in the second result, non-aspect score parts stay fixed, and at least one item changes rank.

- [x] **Step 3: Run the tests and confirm the ranking functions are absent.**

  Run `python3 -m pytest tests/unit/test_demo.py -q`.

- [x] **Step 4: Implement stable ranking and the priority formula.**

  Keep stored non-aspect contributions fixed. Compute the weighted aspect score from learned weights plus temporary priorities, normalize effective weights to one, replace only the aspect contribution, and sort by descending adjusted total then item ID. Reject unknown users and models and clamp K to the available candidate count.

- [x] **Step 5: Run the ranking tests and confirm they pass.**

  Run `python3 -m pytest tests/unit/test_demo.py -q`.

### Task 4: Implement evidence display records and fallback behavior

**Files:**
- Modify: `app/demo.py`
- Modify: `tests/unit/test_demo.py`

**Interfaces:**
- `evidence_for_item(bundle: DemoBundle, item_id: str) -> tuple[DemoEvidence, ...]` returns source passages without reviewer identity fields.
- `evidence_state(rows: Sequence[DemoEvidence]) -> str` returns `supported`, `conflicting`, `insufficient_support`, or `unavailable`.

- [x] **Step 1: Add tests for conflict, weak support, unavailable evidence, and identity suppression.**

  Assert that positive and negative rows produce `conflicting`, low-support rows produce `insufficient_support`, no rows produce `unavailable`, and display dictionaries contain no author or reviewer identity key.

- [x] **Step 2: Run the tests and confirm the evidence helpers are absent.**

  Run `python3 -m pytest tests/unit/test_demo.py -q`.

- [x] **Step 3: Implement evidence grouping and state rules.**

  Use the highest support row for a single-direction status. Use both polarity directions for `conflicting`. Keep review IDs, exact text, timestamps, and sentiment in the returned records.

- [x] **Step 4: Run the evidence tests and confirm they pass.**

  Run `python3 -m pytest tests/unit/test_demo.py -q`.

### Task 5: Build the Streamlit five-minute flow

**Files:**
- Modify: `app/streamlit_app.py`
- Modify: `app/README.md`

**Interfaces:**
- The app calls `load_demo_bundle_with_fallback`, `rank_recommendations`, `compare_rankings`, and `evidence_for_item`.
- The view renders the snapshot, lineage, user history, learned weights, temporary priorities, baseline ranking, selected model ranking, rank changes, evidence, and prepared metrics.

- [x] **Step 1: Add a smoke test for the app module surface.**

  Import the module in a subprocess with `python3 -c "import app.streamlit_app"` and assert that the command exits successfully after the service exists.

- [x] **Step 2: Replace the shell with controls and render functions.**

  Add sidebar controls for snapshot, user, model, K, and aspect priorities. Render separate baseline and selected model tables. Add evidence expanders and a lineage and metric panel. Use a visible fallback warning when external loading fails.

- [x] **Step 3: Run the smoke import and start the app with the fixture.**

  Run `python3 -c "import app.streamlit_app"` and `streamlit run app/streamlit_app.py --server.headless true` with a bounded smoke timeout. Confirm that the app starts without a model fit or network request.

- [x] **Step 4: Update the app README.**

  Document the environment variable, fallback behavior, presenter sequence, prepared-artifact rule, and the exact commands for the demo.

### Task 6: Record T5.2 evidence and update task indexes

**Files:**
- Create: `docs/tasks/t5_2-streamlit-demo.md`
- Modify: `docs/tasks/backlog.md`
- Modify: `docs/tasks/task-matrix.md`
- Modify: `docs/reference/project-status.md`

**Interfaces:**
- The task record links the T5.2 spec, serving contract, fallback fixture hash, source snapshot and cutoff, validation commands, and known fallback limits.

- [x] **Step 1: Compute the fallback fixture SHA-256 and record the source lineage.**

  Use the full snapshot ID and training cutoff from the existing T4.2 and T4.1 records. Record that the fallback is synthetic and does not support production metric claims.

- [x] **Step 2: Write the task record and update the indexes.**

  Mark T5.2 done only after all validation commands pass. Link the app, service, fixture, spec, and task record.

- [x] **Step 3: Run the focused checks.**

  Run `ruff format --check app tests/unit/test_demo.py`, `ruff check app tests/unit/test_demo.py`, `python3 -m pytest tests/unit/test_demo.py -q`, and `python3 -c "import app.streamlit_app"`.

- [x] **Step 4: Run the full verification gate.**

  Run `make check` and `make validate`. Read the complete output and record the exit codes and test count in the task record.

- [x] **Step 5: Review the diff and preserve unrelated work.**

  Run `git diff --check` and `git status --short`. Make sure that only T5.2 files are staged for the final commit. Leave existing T5.1 working-tree changes untouched.
