import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.tune_bpr import tune_bpr

CUTOFF = "1970-01-01T00:00:00.100000+00:00"


def _write_snapshot(tmp_path: Path) -> Path:
    train = [
        {
            "review_id": f"r{i}",
            "user_id": f"u{i % 30:02d}",
            "item_id": f"i{(i * 7) % 19:02d}",
            "rating": float(1 + i % 5),
            "timestamp": i % 90,
        }
        for i in range(200)
    ]
    targets = [
        {
            "review_id": f"t{i}",
            "user_id": f"u{i % 30:02d}",
            "item_id": f"i{(i * 5) % 19:02d}",
            "rating": 5.0,
            "timestamp": 100 + i,
        }
        for i in range(40)
    ]
    pq.write_table(pa.Table.from_pylist(train), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(targets), tmp_path / "validation.parquet")
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": CUTOFF},
                "artifact_paths": {
                    "train_interactions": "train.parquet",
                    "validation_targets": "validation.parquet",
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_tune_bpr_records_every_trial_and_saves_the_best(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    result = tune_bpr(
        manifest,
        output_path=tmp_path / "tuning.json",
        learning_rates=(0.01, 0.1),
        regularizations=(0.001,),
        epochs=(3,),
        embedding_dimensions=(8,),
        bootstrap_samples=20,
        save_best_path=tmp_path / "best.npz",
    )

    assert result["status"] == "complete"
    assert result["tuning_budget"]["trials"] == 2
    assert len(result["trials"]) == 2
    best_score = result["best"]["metrics"]["ndcg@10"]["estimate"]
    assert best_score == max(t["metrics"]["ndcg@10"]["estimate"] for t in result["trials"])
    assert result["best_model_file"]["sha256"]
    with pytest.raises(FileExistsError, match="completed"):
        tune_bpr(manifest, output_path=tmp_path / "tuning.json", epochs=(3,))


def test_tune_bpr_refuses_the_test_cutoff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="t0"):
        tune_bpr(_write_snapshot(tmp_path), output_path=tmp_path / "x.json", cutoff_name="t1")


@pytest.mark.parametrize(
    ("model_id", "grid"),
    [
        ("b1_item_knn", [{"n_neighbors": None}, {"n_neighbors": 2}]),
        ("b3_ppr", [{"damping": 0.5, "max_iter": 5}, {"damping": 0.85, "max_iter": 10}]),
    ],
)
def test_tune_baseline_supports_knn_and_ppr(tmp_path: Path, model_id: str, grid) -> None:
    from scripts.tune_baseline import tune_baseline

    result = tune_baseline(
        _write_snapshot(tmp_path),
        model_id=model_id,
        output_path=tmp_path / "tuning.json",
        grid=grid,
        bootstrap_samples=20,
    )

    assert result["model_id"] == model_id
    assert result["tuning_budget"]["trials"] == 2
    assert result["grid"] == grid
