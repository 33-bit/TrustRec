import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.run_ranking_baselines import run_baseline


def test_run_baseline_writes_snapshot_and_model_lineage(tmp_path: Path) -> None:
    interactions = tmp_path / "train_interactions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"user_id": "u1", "item_id": "h", "rating": 5.0, "timestamp": 1},
                {"user_id": "u2", "item_id": "h", "rating": 5.0, "timestamp": 2},
                {"user_id": "u2", "item_id": "a", "rating": 5.0, "timestamp": 3},
                {"user_id": "u3", "item_id": "b", "rating": 5.0, "timestamp": 4},
            ]
        ),
        interactions,
    )
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": "1970-01-01T00:00:00.010000+00:00"},
                "artifact_paths": {"train_interactions": interactions.name},
            }
        )
    )
    output = tmp_path / "ranking.json"

    artifact = run_baseline(
        manifest,
        model_id="b0_most_popular",
        user_id="u1",
        cutoff_name="t0",
        k=2,
        output_path=output,
    )

    assert artifact["snapshot_id"] == "snap-1"
    assert artifact["dataset_hash"] == "d" * 64
    assert artifact["model_id"] == "b0_most_popular"
    assert artifact["status"] == "complete"
    assert artifact["seed"] is None
    assert artifact["configuration_hash"]
    assert artifact["source_artifact_sha256"]
    assert artifact["result"]["items"][0]["item_id"] == "a"
    assert json.loads(output.read_text()) == artifact
