from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.run_evaluation import run_evaluation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rankings(path: Path, *, ranked: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "user_id": "u1",
                "snapshot_id": "snap-1",
                "cutoff_timestamp": "1970-01-01T00:00:00.010000+00:00",
                "candidate_item_ids": ["a", "b"],
                "ranked_item_ids": ranked,
                "explanation_coverage": 1.0,
                "latency_ms": 2.4,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_fixture_snapshot(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    interactions = tmp_path / "train_interactions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"review_id": "r1", "user_id": "u1", "item_id": "h", "rating": 5.0, "timestamp": 1},
                {"review_id": "r2", "user_id": "u2", "item_id": "a", "rating": 5.0, "timestamp": 2},
                {"review_id": "r3", "user_id": "u2", "item_id": "b", "rating": 5.0, "timestamp": 3},
            ]
        ),
        interactions,
    )
    targets = tmp_path / "test_targets.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "r4",
                    "user_id": "u1",
                    "item_id": "a",
                    "rating": 5.0,
                    "timestamp": 10,
                },
            ]
        ),
        targets,
    )
    b0 = tmp_path / "b0.jsonl"
    t0 = tmp_path / "t0.jsonl"
    _write_rankings(b0, ranked=["a", "b"])
    _write_rankings(t0, ranked=["b", "a"])
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {
                    "t0": "1970-01-01T00:00:00.005000+00:00",
                    "t1": "1970-01-01T00:00:00.010000+00:00",
                },
                "artifact_paths": {
                    "fit_interactions": interactions.name,
                    "test_targets": targets.name,
                },
                "artifact_sha256": {
                    "fit_interactions": _sha256(interactions),
                    "test_targets": _sha256(targets),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, {"b0": b0, "t0": t0}


def test_run_evaluation_writes_metrics_and_manifest(tmp_path: Path) -> None:
    snapshot_manifest, ranking_paths = write_fixture_snapshot(tmp_path)
    output_path = tmp_path / "metrics.json"
    manifest_path = tmp_path / "run.manifest.json"

    result = run_evaluation(
        snapshot_manifest,
        model_inputs={
            "b0_most_popular": ranking_paths["b0"],
            "t0_trustrec": ranking_paths["t0"],
        },
        output_path=output_path,
        manifest_path=manifest_path,
        run_id="t5-1-fixture",
        cutoff_name="t1",
        bootstrap_samples=20,
    )

    assert result["status"] == "complete"
    assert result["snapshot_id"] == "snap-1"
    assert result["models"]["b0_most_popular"]["metrics"]["ndcg@5"] == 1.0
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["metrics_path"] == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_run_evaluation_does_not_overwrite_completed_manifest(tmp_path: Path) -> None:
    snapshot_manifest, ranking_paths = write_fixture_snapshot(tmp_path)
    run_manifest = tmp_path / "run.manifest.json"
    run_manifest.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    with pytest.raises(FileExistsError, match="completed"):
        run_evaluation(
            snapshot_manifest,
            model_inputs={"b0": ranking_paths["b0"], "t0": ranking_paths["t0"]},
            output_path=tmp_path / "metrics.json",
            manifest_path=run_manifest,
            run_id="t5-1-fixture",
            cutoff_name="t1",
            bootstrap_samples=20,
        )
