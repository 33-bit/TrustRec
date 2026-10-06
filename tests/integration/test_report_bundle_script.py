from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


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
                "status": "complete",
                "run_id": "t5-1-fixture",
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "protocol_version": "evaluation-v1",
                "model_seeds": [7],
                "candidate_rule": "known_items_before_cutoff_minus_user_history",
                "target_rule": "first_new_item_with_rating_at_least_4",
                "evaluation": {"candidate_set_hash": "c" * 64, "eligible_users": 2},
                "models": {
                    "b0": {
                        "metrics": {"ndcg@10": 0.5},
                        "counts": {"eligible_users": 2},
                        "metadata": {"candidate_set_hash": "c" * 64},
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
                "run_id": "t5-1-fixture",
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "protocol_version": "evaluation-v1",
                "seed": 7,
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
                    },
                    {
                        "artifact_id": "run_manifest",
                        "path": "run.manifest.json",
                        "sha256": _sha256(manifest),
                    },
                ],
                "commands": [
                    {
                        "command_id": "evaluate_fixture",
                        "command": "python3 run_fixture.py",
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
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return spec


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "scripts/package_report.py", *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_package_report_cli_creates_and_verifies_saved_bundle(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    spec = _write_fixture(fixture_root)
    output = tmp_path / "bundle"

    created = _run(
        "--spec",
        str(spec),
        "--output-dir",
        str(output),
        "--root",
        str(fixture_root),
        "--code-revision",
        "fixture",
    )
    assert created.returncode == 0, created.stderr
    verified = _run("--verify-bundle", str(output))
    assert verified.returncode == 0, verified.stderr

    (output / "report.md").write_text("changed\n", encoding="utf-8")
    rejected = _run("--verify-bundle", str(output))
    assert rejected.returncode == 2
    assert "does not match" in rejected.stderr
