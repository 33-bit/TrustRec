from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.toml"
    config.write_text("seed = 7\n", encoding="utf-8")
    metrics = root / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": "t5-1-fixture",
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": "1970-01-01T00:00:00Z", "t1": "1970-01-02T00:00:00Z"},
                "candidate_rule": "known_items_before_cutoff_minus_user_history",
                "target_rule": "first_new_item_with_rating_at_least_4",
                "model_seeds": [7],
                "protocol_version": "evaluation-v1",
                "cutoff_name": "t1",
                "cutoff_timestamp": "1970-01-02T00:00:00Z",
                "bootstrap": {
                    "method": "paired_user_bootstrap",
                    "confidence_level": 0.95,
                    "samples": 20,
                },
                "evaluation": {
                    "candidate_set_hash": "c" * 64,
                    "eligible_users": 2,
                    "exclusions": {"no_candidates": 1},
                },
                "models": {
                    "b0": {
                        "metrics": {"ndcg@10": 0.5},
                        "bootstrap": {
                            "ndcg@10": {
                                "estimate": 0.5,
                                "lower": 0.25,
                                "upper": 0.75,
                            }
                        },
                        "counts": {"eligible_users": 2},
                        "metadata": {"candidate_set_hash": "c" * 64},
                        "slices": {"history": {"1-2": {"ndcg@10": 0.4}}},
                        "resource": {"latency_ms_mean": 2.5, "memory_bytes_mean": 12.0},
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = root / "run.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "T5.1",
                "run_id": "t5-1-fixture",
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "protocol_version": "evaluation-v1",
                "seed": 7,
                "seeds": [7],
                "cutoffs": {"t0": "1970-01-01T00:00:00Z", "t1": "1970-01-02T00:00:00Z"},
                "metrics_path": "metrics.json",
                "artifact_sha256": {"metrics": _sha256(metrics)},
                "status": "complete",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    spec = root / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_id": "fixture-bundle",
                "title": "Fixture report",
                "environment_files": [
                    {"artifact_id": "config", "path": "config.toml", "sha256": _sha256(config)}
                ],
                "artifacts": [
                    {
                        "artifact_id": "metrics",
                        "path": "metrics.json",
                        "sha256": _sha256(metrics),
                        "availability": "required",
                    },
                    {
                        "artifact_id": "run_manifest",
                        "path": "run.manifest.json",
                        "sha256": _sha256(manifest),
                        "availability": "required",
                    },
                ],
                "commands": [
                    {
                        "command_id": "evaluate_fixture",
                        "command": "python3 run_fixture.py",
                        "prerequisites": ["Run from the repository root."],
                    }
                ],
                "runs": [
                    {
                        "run_id": "t5-1-fixture",
                        "manifest_artifact": "run_manifest",
                        "metrics_artifact": "metrics",
                        "command_id": "evaluate_fixture",
                        "evidence_role": "recommendation_test",
                    }
                ],
                "claims": [
                    {
                        "claim_id": "fixture-ndcg",
                        "kind": "measured",
                        "statement": "The fixture NDCG@10 estimate is 0.5.",
                        "run_id": "t5-1-fixture",
                        "model_id": "b0",
                        "metric": "ndcg@10",
                        "evidence": [
                            {
                                "artifact_id": "metrics",
                                "label": "NDCG@10 estimate",
                                "pointer": "/models/b0/metrics/ndcg@10",
                            }
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return spec


def test_package_report_resolves_claims_and_writes_reproduction_links(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    manifest = package_report(spec, output, repository_root=tmp_path, code_revision="fixture")

    assert manifest["status"] == "complete"
    assert manifest["claims"][0]["evidence"][0]["value"] == 0.5
    assert manifest["claims"][0]["evidence"][0]["run_id"] == "t5-1-fixture"
    assert "bundle.manifest.json" in (output / "report.md").read_text(encoding="utf-8")
    assert "commands.md#evaluate_fixture" in (output / "report.md").read_text(encoding="utf-8")
    assert (output / "sources" / "run_manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert not (output / "raw").exists()

    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["value"] == "0.5"
    assert row["cutoff"] == "1970-01-02T00:00:00Z"
    assert row["eligible_users"] == "2"
    assert row["ci_lower"] == "0.25"
    assert row["ci_upper"] == "0.75"
    assert row["uncertainty_method"] == "paired_user_bootstrap"
    assert json.loads(row["slices"]) == {"history": {"1-2": {"ndcg@10": 0.4}}}
    assert json.loads(row["resource"]) == {"latency_ms_mean": 2.5, "memory_bytes_mean": 12.0}
    assert json.loads(row["exclusions"]) == {"no_candidates": 1}


def test_package_report_is_deterministic_for_the_same_inputs(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    package_report(spec, first, repository_root=tmp_path, code_revision="fixture")
    package_report(spec, second, repository_root=tmp_path, code_revision="fixture")

    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    assert all(
        (first / relative).read_bytes() == (second / relative).read_bytes()
        for relative in first_files
    )


def test_package_report_rejects_hash_changes_and_completed_output(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    (tmp_path / "metrics.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        package_report(spec, output, repository_root=tmp_path, code_revision="fixture")

    spec = _write_fixture(tmp_path / "fresh")
    output = tmp_path / "fresh-output"
    package_report(spec, output, repository_root=tmp_path / "fresh", code_revision="fixture")
    with pytest.raises(FileExistsError, match="already exists"):
        package_report(spec, output, repository_root=tmp_path / "fresh", code_revision="fixture")


def test_package_report_rejects_path_traversal_and_duplicate_claim_ids(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = "../metrics.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="repository root"):
        package_report(spec, tmp_path / "bundle", repository_root=tmp_path, code_revision="fixture")

    spec = _write_fixture(tmp_path / "duplicate")
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["claims"].append(dict(payload["claims"][0]))
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="claim_id"):
        package_report(
            spec,
            tmp_path / "duplicate-output",
            repository_root=tmp_path / "duplicate",
            code_revision="fixture",
        )


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("duplicate_run", "run_id"),
        ("unknown_command", "command"),
        ("empty_command", "command"),
        ("incomplete_run", "complete"),
        ("manifest_run_mismatch", "run_id"),
        ("metrics_seed_mismatch", "seeds"),
        ("metrics_cutoff_mismatch", "cutoffs"),
        ("missing_metrics_hash", "metrics.*hash"),
        ("metrics_path_mismatch", "metrics_path"),
        ("absolute_metrics_path", "relative path"),
        ("parent_metrics_path", "repository root"),
        ("candidate_hash_mismatch", "candidate"),
        ("top_level_count_mismatch", "eligible-user"),
        ("missing_model_count", "eligible"),
        ("synthetic_measurement", "synthetic"),
        ("design_measurement", "design"),
        ("unrelated_evidence", "run"),
        ("bad_pointer", "pointer"),
        ("non_scalar", "scalar"),
        ("nan_value", "finite|JSON"),
    ],
)
def test_package_report_rejects_broken_claim_lineage(tmp_path: Path, case: str, error: str) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    payload = json.loads(spec.read_text())
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    manifest_path = tmp_path / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "duplicate_run":
        payload["runs"].append(dict(payload["runs"][0]))
    elif case == "unknown_command":
        payload["runs"][0]["command_id"] = "missing"
    elif case == "empty_command":
        payload["commands"][0]["command"] = ""
    elif case == "incomplete_run":
        manifest["status"] = "failed"
    elif case == "manifest_run_mismatch":
        manifest["run_id"] = "different-run"
    elif case == "metrics_seed_mismatch":
        metrics["model_seeds"] = [17]
    elif case == "metrics_cutoff_mismatch":
        metrics["cutoffs"]["t1"] = "1970-01-03T00:00:00Z"
    elif case == "missing_metrics_hash":
        manifest["artifact_sha256"] = {}
    elif case == "metrics_path_mismatch":
        manifest["metrics_path"] = "other-metrics.json"
    elif case == "absolute_metrics_path":
        manifest["metrics_path"] = str(metrics_path)
    elif case == "parent_metrics_path":
        manifest["metrics_path"] = "../metrics.json"
    elif case == "candidate_hash_mismatch":
        metrics["models"]["b0"]["metadata"]["candidate_set_hash"] = "b" * 64
    elif case == "top_level_count_mismatch":
        metrics["evaluation"]["eligible_users"] = 3
    elif case == "missing_model_count":
        metrics["models"]["b0"].pop("counts")
    elif case == "synthetic_measurement":
        payload["runs"][0]["evidence_role"] = "synthetic"
    elif case == "design_measurement":
        payload["runs"][0]["evidence_role"] = "design"
    elif case == "unrelated_evidence":
        unrelated_path = tmp_path / "unrelated.json"
        unrelated_path.write_text('{"value": 1.0}')
        payload["artifacts"].append(
            {
                "artifact_id": "unrelated",
                "path": "unrelated.json",
                "sha256": _sha256(unrelated_path),
            }
        )
        payload["claims"][0]["evidence"][0].update(artifact_id="unrelated", pointer="/value")
    elif case == "bad_pointer":
        payload["claims"][0]["evidence"][0]["pointer"] = "/absent"
    elif case == "non_scalar":
        payload["claims"][0]["evidence"][0]["pointer"] = "/models"
    elif case == "nan_value":
        metrics["models"]["b0"]["metrics"]["ndcg@10"] = float("nan")
    metrics_path.write_text(json.dumps(metrics))
    if case != "missing_metrics_hash":
        manifest["artifact_sha256"]["metrics"] = _sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest))
    for record in payload["artifacts"]:
        record["sha256"] = _sha256(tmp_path / record["path"])
    spec.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=error):
        package_report(spec, tmp_path / "bundle", repository_root=tmp_path, code_revision="fixture")
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-*"))


def test_package_report_records_missing_external_artifacts_without_inventing_values(
    tmp_path: Path,
) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    payload = json.loads(spec.read_text())
    payload["artifacts"].append(
        {
            "artifact_id": "raw_table",
            "path": "data/processed/interactions.parquet",
            "sha256": "e" * 64,
            "availability": "external",
        }
    )
    spec.write_text(json.dumps(payload))

    result = package_report(
        spec, tmp_path / "bundle", repository_root=tmp_path, code_revision="fixture"
    )

    assert result["sources"]["raw_table"]["exists"] is False
    assert result["sources"]["raw_table"]["expected_sha256"] == "e" * 64


def test_package_report_preserves_frozen_llm_provenance(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    payload = json.loads(spec.read_text())
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["pseudo_test_label_provenance"] = {
        "model_id": "gpt-6.1-sol",
        "model_revision": "revision-1",
        "prompt_version": "t2.2-llm-v1",
        "temperature": "unavailable",
        "seed": "unavailable",
    }
    metrics_path.write_text(json.dumps(metrics))
    manifest_path = tmp_path / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"frozen_pseudo_test": True, "pseudo_test_tuning_allowed": False})
    manifest["artifact_sha256"]["metrics"] = _sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest))
    for record in payload["artifacts"]:
        record["sha256"] = _sha256(tmp_path / record["path"])
    payload["runs"][0]["evidence_role"] = "llm_pseudo_test"
    spec.write_text(json.dumps(payload))

    output = tmp_path / "bundle"
    package_report(spec, output, repository_root=tmp_path, code_revision="fixture")

    claims = json.loads((output / "claims.json").read_text())
    run = claims["claims"][0]["run"]
    assert run["llm_provenance"] == metrics["pseudo_test_label_provenance"]
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["llm_model_id"] == "gpt-6.1-sol"
    assert row["llm_prompt_version"] == "t2.2-llm-v1"
    assert row["llm_model_revision"] == "revision-1"
    assert row["llm_seed"] == "unavailable"


def test_verify_bundle_rejects_source_path_outside_bundle(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report, verify_bundle

    spec = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    package_report(spec, output, repository_root=tmp_path, code_revision="fixture")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    manifest_path = output / "bundle.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"]["run_manifest"]["bundle_path"] = "../outside.txt"
    manifest["sources"]["run_manifest"]["sha256"] = _sha256(outside)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="inside the bundle"):
        verify_bundle(output)


def test_bundle_verification_requires_a_complete_checksum_inventory(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report, verify_bundle

    spec = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    package_report(spec, output, repository_root=tmp_path, code_revision="fixture")
    (output / "SHA256SUMS").write_text("")

    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(output)


def test_package_report_rejects_symlinked_source_paths(tmp_path: Path) -> None:
    from trustrec.reporting.bundle import package_report

    spec = _write_fixture(tmp_path)
    link = tmp_path / "linked-config.toml"
    link.symlink_to(tmp_path / "config.toml")
    payload = json.loads(spec.read_text())
    payload["environment_files"][0]["path"] = "linked-config.toml"
    payload["environment_files"][0]["sha256"] = _sha256(tmp_path / "config.toml")
    spec.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="symbolic link"):
        package_report(spec, tmp_path / "bundle", repository_root=tmp_path, code_revision="fixture")
