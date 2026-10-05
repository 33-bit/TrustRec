# T5.1 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a snapshot-safe evaluator that compares prepared baseline and ablation rankings with versioned metrics, paired bootstrap intervals, sparse-history slices, and lineage manifests.

**Architecture:** `metrics.py` contains pure ranking metrics and bootstrap summaries. `protocol.py` builds one locked evaluation case per target user, validates ranking rows against those cases, assigns history and popularity slices, and compares model outputs. `scripts/run_evaluation.py` handles Parquet and JSON Lines I/O, hashes inputs, and writes metrics and run-manifest artifacts.

**Tech Stack:** Python 3.11+, NumPy, PyArrow, JSON Lines, pytest, and Ruff.

**Spec:** `docs/superpowers/specs/2026-10-05-t5-1-evaluation-design.md`

## Global Constraints

- Use the `global-temporal-v1` protocol from `docs/specs/evaluation-contract.md`.
- Use `T0 < T1`, train or fit rows strictly before the selected cutoff, and recommendation targets at or after that cutoff.
- Use the positive target rating threshold `4`.
- Use K values `5`, `10`, and `20`.
- Use paired user bootstrap with seeds `7`, `17`, and `27`, 2,000 samples, and a 0.95 confidence level by default.
- Require equal users, candidate IDs, and target IDs for every compared model.
- Do not use frozen `llm_pseudo_test` labels as recommendation targets or features.
- Do not download source data or write generated full-data artifacts to Git.
- Preserve JSON hashes and reject replacement of a completed manifest.

### Task 1: Implement pure ranking metrics and paired bootstrap

**Files:**
- Create: `src/trustrec/evaluation/metrics.py`
- Create: `tests/unit/test_evaluation_metrics.py`

**Interfaces:**
- Produces `ndcg_at_k(ranked_item_ids: Sequence[str], target_item_ids: Collection[str], k: int) -> float`.
- Produces `recall_at_k(ranked_item_ids: Sequence[str], target_item_ids: Collection[str], k: int) -> float`.
- Produces `precision_at_k(ranked_item_ids: Sequence[str], target_item_ids: Collection[str], k: int) -> float`.
- Produces `intra_list_diversity(ranked_item_ids: Sequence[str], item_aspects: Mapping[str, Collection[str]], k: int) -> float | None`.
- Produces `paired_user_bootstrap(values_by_model: Mapping[str, Sequence[float]], *, seed: int, samples: int, confidence_level: float, reference_model: str | None = None) -> dict[str, BootstrapSummary]`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_binary_ranking_metrics_use_top_k_and_target_count():
    ranked = ["b", "a", "c"]
    targets = {"a", "c"}
    assert recall_at_k(ranked, targets, 2) == 0.5
    assert precision_at_k(ranked, targets, 2) == 0.5
    assert ndcg_at_k(ranked, targets, 2) == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-module failure**

Run: `python3 -m pytest tests/unit/test_evaluation_metrics.py -q`

Expected: FAIL because `trustrec.evaluation.metrics` does not exist.

- [ ] **Step 3: Add minimal metric implementations**

Implement finite positive K validation, duplicate ranked-item rejection, binary relevance, zero for empty target sets, and average pairwise Jaccard diversity. Return `None` for diversity when no ranked item has an aspect set.

- [ ] **Step 4: Add failing bootstrap tests**

```python
def test_paired_bootstrap_reports_reproducible_model_delta():
    result = paired_user_bootstrap(
        {"b0": [0.0, 1.0, 1.0], "t0": [1.0, 1.0, 1.0]},
        seed=7,
        samples=200,
        confidence_level=0.95,
        reference_model="b0",
    )
    assert result["t0"].estimate == pytest.approx(1.0)
    assert result["t0"].delta_estimate == pytest.approx(1 / 3)
    assert result == paired_user_bootstrap(
        {"b0": [0.0, 1.0, 1.0], "t0": [1.0, 1.0, 1.0]},
        seed=7,
        samples=200,
        confidence_level=0.95,
        reference_model="b0",
    )
```

- [ ] **Step 5: Run the bootstrap test and verify it fails for the missing function**

Run: `python3 -m pytest tests/unit/test_evaluation_metrics.py::test_paired_bootstrap_reports_reproducible_model_delta -q`

Expected: FAIL because `paired_user_bootstrap` is not implemented.

- [ ] **Step 6: Implement bounded-memory paired resampling**

Use `numpy.random.default_rng(seed)`, resample the same user indices for every model, compute percentile limits from the replicate means, and include `estimate`, `lower`, `upper`, `samples`, `seed`, `confidence_level`, and optional paired delta fields in an immutable `BootstrapSummary`.

- [ ] **Step 7: Run the focused tests and format the module**

Run: `python3 -m pytest tests/unit/test_evaluation_metrics.py -q`

Expected: PASS with all metric and bootstrap tests green.

Run: `ruff format src/trustrec/evaluation/metrics.py tests/unit/test_evaluation_metrics.py && ruff check src/trustrec/evaluation/metrics.py tests/unit/test_evaluation_metrics.py`

Expected: exit code 0.

### Task 2: Build temporal evaluation cases and model comparison protocol

**Files:**
- Create: `src/trustrec/evaluation/protocol.py`
- Modify: `src/trustrec/evaluation/__init__.py`
- Create: `tests/unit/test_evaluation_protocol.py`

**Interfaces:**
- Produces `EvaluationCase` with `user_id`, `candidate_item_ids`, `target_item_ids`, `history_count`, `history_slice`, and `target_popularity_slices`.
- Produces `EvaluationDataset` with `snapshot_id`, `cutoff_timestamp`, `cases`, `exclusions`, `candidate_set_hash`, and `popularity_counts`.
- Consumes ranking rows with `user_id`, `snapshot_id`, `cutoff_timestamp`, `candidate_item_ids`, and `ranked_item_ids` plus optional explanation and latency fields.
- Produces `ModelEvaluation` with aggregate metrics, per-user metrics, slice counts, and resource summaries.
- Produces `compare_model_rankings(dataset: EvaluationDataset, model_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, k_values: Sequence[int], item_aspects: Mapping[str, Collection[str]] | None, bootstrap_seed: int, bootstrap_samples: int, confidence_level: float, reference_model: str | None) -> dict[str, ModelEvaluation]`.

- [ ] **Step 1: Write failing case-building tests**

```python
def test_build_evaluation_cases_removes_history_and_future_targets():
    dataset = build_evaluation_cases(
        interactions=[
            {"review_id": "r1", "user_id": "u1", "item_id": "seen", "rating": 2, "timestamp": 1},
            {
                "review_id": "r2",
                "user_id": "u2",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 2,
            },
        ],
        targets=[
            {
                "review_id": "r3",
                "user_id": "u1",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 10,
            },
            {
                "review_id": "r4",
                "user_id": "u1",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 9,
            },
        ],
        snapshot_id="snap-1",
        cutoff_timestamp=10,
    )
    assert dataset.cases[0].candidate_item_ids == ("candidate",)
    assert dataset.cases[0].target_item_ids == frozenset({"candidate"})
    assert dataset.exclusions["target_before_cutoff"] == 1
```

- [ ] **Step 2: Run the case test and verify the expected missing-module failure**

Run: `python3 -m pytest tests/unit/test_evaluation_protocol.py::test_build_evaluation_cases_removes_history_and_future_targets -q`

Expected: FAIL because `trustrec.evaluation.protocol` does not exist.

- [ ] **Step 3: Implement strict case construction**

Normalize interactions with `first_interactions_before_cutoff`, reject malformed IDs and timestamps, derive known catalog and history, keep only target rows at or after the cutoff with rating at least 4, record exclusion reasons, and hash sorted user and candidate records with `stable_model_hash`.

- [ ] **Step 4: Add ranking validation tests**

```python
def test_compare_model_rankings_rejects_candidate_set_mismatch():
    dataset = fixture_dataset()
    rows = {"b0": fixture_rows(dataset), "t0": fixture_rows(dataset, candidate_item_ids=["wrong"])}
    with pytest.raises(LeakageError, match="candidate"):
        compare_model_rankings(
            dataset,
            rows,
            k_values=(5,),
            bootstrap_seed=7,
            bootstrap_samples=20,
            confidence_level=0.95,
        )
```

- [ ] **Step 5: Implement ranking validation and metric aggregation**

Require exactly one row per case for every model. Validate snapshot and cutoff, exact candidate IDs, ranked-item uniqueness and membership, optional explanation values in `[0, 1]`, and non-negative latency. Compute metrics for all K values, coverage, diversity, explanation coverage, candidate retention, latency, history slices, and target popularity slices. Store explicit counts for eligible users, evaluated users, target items, and exclusions.

- [ ] **Step 6: Add comparison bootstrap tests**

```python
def test_compare_model_rankings_uses_shared_user_bootstrap():
    dataset = fixture_dataset()
    results = compare_model_rankings(
        dataset,
        {"b0": fixture_rows(dataset), "t0": fixture_rows(dataset, reverse=True)},
        k_values=(1,),
        bootstrap_seed=7,
        bootstrap_samples=100,
        confidence_level=0.95,
        reference_model="b0",
    )
    assert results["t0"].bootstrap["ndcg@1"].delta_estimate < 0
```

- [ ] **Step 7: Run focused protocol tests and format the module**

Run: `python3 -m pytest tests/unit/test_evaluation_protocol.py -q`

Expected: PASS.

Run: `ruff format src/trustrec/evaluation/protocol.py src/trustrec/evaluation/__init__.py tests/unit/test_evaluation_protocol.py && ruff check src/trustrec/evaluation/protocol.py src/trustrec/evaluation/__init__.py tests/unit/test_evaluation_protocol.py`

Expected: exit code 0.

### Task 3: Add the reproducible evaluation runner and manifest writer

**Files:**
- Create: `scripts/run_evaluation.py`
- Create: `tests/integration/test_evaluation_script.py`
- Modify: `src/trustrec/evaluation/protocol.py` only if the integration contract exposes a missing validated field

**Interfaces:**
- Produces `run_evaluation(snapshot_manifest_path: Path, *, model_inputs: Mapping[str, Path], output_path: Path, manifest_path: Path, run_id: str, cutoff_name: str, target_path: Path | None = None, config_path: Path = Path("configs/evaluation.toml"), k_values: Sequence[int] = (5, 10, 20), positive_threshold: float = 4.0, bootstrap_seed: int = 7, bootstrap_samples: int = 2000, confidence_level: float = 0.95, item_aspects_path: Path | None = None, reference_model: str | None = None) -> dict[str, Any]`.
- Consumes `MODEL_ID=PATH` arguments from the CLI and writes a metrics JSON and a manifest JSON.

- [ ] **Step 1: Write a failing integration test for lineage and output**

```python
def test_run_evaluation_writes_metrics_and_manifest(tmp_path: Path):
    manifest, ranking_paths = write_fixture_snapshot(tmp_path)
    result = run_evaluation(
        manifest,
        model_inputs={"b0_most_popular": ranking_paths["b0"], "t0_trustrec": ranking_paths["t0"]},
        output_path=tmp_path / "metrics.json",
        manifest_path=tmp_path / "run.manifest.json",
        run_id="t5-1-fixture",
        cutoff_name="t1",
        bootstrap_samples=20,
    )
    assert result["status"] == "complete"
    assert result["snapshot_id"] == "snap-1"
    assert json.loads((tmp_path / "run.manifest.json").read_text())["metrics_path"] == str(
        tmp_path / "metrics.json"
    )
```

- [ ] **Step 2: Run the integration test and verify the expected missing-script failure**

Run: `python3 -m pytest tests/integration/test_evaluation_script.py -q`

Expected: FAIL because `scripts.run_evaluation` does not exist.

- [ ] **Step 3: Implement manifest and input loaders**

Load the snapshot manifest, resolve relative artifact paths from the repository root and manifest directory, hash source and ranking files, load JSON Lines with line-number errors, and reject completed manifest paths before any output write. Read `validation_targets` for `t0` and `test_targets` for `t1` when `--targets` is absent.

- [ ] **Step 4: Implement the runner and JSON output**

Call `build_evaluation_cases` and `compare_model_rankings`. Write a metrics object containing protocol version, run ID, snapshot and dataset hashes, cutoff, model IDs, candidate and target rules, metric results, bootstrap configuration, slice counts, and source artifact hashes. Write a manifest containing `run_id`, `snapshot_id`, `protocol_version`, `config_path`, `config_sha256`, `dataset_hash`, `cutoffs`, `seed`, `metrics_path`, `artifact_sha256`, and `status`.

- [ ] **Step 5: Add the CLI and no-overwrite test**

Support `--snapshot-manifest`, repeated `--model MODEL_ID=PATH`, `--cutoff`, `--targets`, `--output`, `--manifest`, `--run-id`, `--config`, `--bootstrap-seed`, `--bootstrap-samples`, `--confidence-level`, `--positive-threshold`, `--reference-model`, and `--item-aspects`. Reject malformed model assignments and completed manifest paths.

```python
def test_run_evaluation_does_not_overwrite_completed_manifest(tmp_path: Path):
    manifest, ranking_paths = write_fixture_snapshot(tmp_path)
    run_manifest = tmp_path / "run.manifest.json"
    run_manifest.write_text(json.dumps({"status": "complete"}))
    with pytest.raises(FileExistsError, match="completed"):
        run_evaluation(
            manifest,
            model_inputs={"b0": ranking_paths["b0"], "t0": ranking_paths["t0"]},
            output_path=tmp_path / "metrics.json",
            manifest_path=run_manifest,
            run_id="t5-1-fixture",
            cutoff_name="t1",
            bootstrap_samples=20,
        )
```

- [ ] **Step 6: Run integration tests and format the runner**

Run: `python3 -m pytest tests/integration/test_evaluation_script.py -q`

Expected: PASS.

Run: `ruff format scripts/run_evaluation.py tests/integration/test_evaluation_script.py && ruff check scripts/run_evaluation.py tests/integration/test_evaluation_script.py`

Expected: exit code 0.

### Task 4: Record the task contract and repository validation

**Files:**
- Create: `docs/tasks/t5_1-evaluation.md`
- Modify: `configs/evaluation.toml`
- Modify: `scripts/README.md`
- Modify: `docs/reference/project-status.md`
- Modify: `docs/tasks/backlog.md`
- Modify: `docs/tasks/task-matrix.md`
- Modify: `scripts/validate_repo.py`
- Modify: `src/trustrec/evaluation/README.md`

**Interfaces:**
- Documents the command, artifact fields, exclusion rules, and fixture validation.
- Keeps the default protocol values aligned with `configs/evaluation.toml`.

- [ ] **Step 1: Extend evaluation configuration with explicit slice and artifact settings**

Add `protocol_version = "evaluation-v1"`, `target_split = "recommendation_test"`, `candidate_rule = "known_items_before_cutoff_minus_user_history"`, `positive_target_rating = 4`, and `ranking_artifact_format = "jsonl_per_user"` without changing existing seed or bootstrap values.

- [ ] **Step 2: Write the T5.1 task record**

Record the prepared-ranking input shape, full-data command, metrics and slice definitions, manifest lineage, no-overwrite rule, and the fact that full generated data remains external.

- [ ] **Step 3: Update repository indexes and status**

Mark T5.1 done only after the verification commands pass. Link the task record from the backlog and task matrix, add it to `scripts/validate_repo.py`, and state that full-category metrics remain pending until external ranking artifacts are supplied.

- [ ] **Step 4: Run documentation and repository checks**

Run: `python3 scripts/validate_repo.py`

Expected: exit code 0 and the new task record is present.

Run: `ruff format --check src/trustrec/evaluation scripts/run_evaluation.py tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_protocol.py tests/integration/test_evaluation_script.py`

Expected: exit code 0.

### Task 5: Run the complete verification suite

**Files:**
- Modify: none unless a verification failure identifies a concrete defect

- [ ] **Step 1: Run focused evaluation tests**

Run: `python3 -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation_protocol.py tests/integration/test_evaluation_script.py -q`

Expected: all evaluation tests pass.

- [ ] **Step 2: Run the complete test suite**

Run: `python3 -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Run the complete lint and format gates**

Run: `ruff format --check . && ruff check .`

Expected: exit code 0.

- [ ] **Step 4: Inspect the final diff and status**

Run: `git diff --check && git status --short`

Expected: no whitespace errors, and only the T5.1 files are changed.
