import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.run_evaluation import run_evaluation
from scripts.run_rankings_batch import run_rankings_batch

from trustrec.recommenders.registry import BASELINE_MODEL_IDS, BaselineParameters

CUTOFF = "1970-01-01T00:00:00.100000+00:00"


def _write_snapshot(tmp_path: Path) -> Path:
    interactions = []
    for position in range(240):
        interactions.append(
            {
                "review_id": f"r{position}",
                "user_id": f"u{position % 40:02d}",
                "item_id": f"i{(position * 7) % 23:02d}",
                "rating": float(1 + position % 5),
                "timestamp": position % 90,
            }
        )
    targets = [
        {
            "review_id": f"t{position}",
            "user_id": f"u{position % 45:02d}",
            "item_id": f"i{(position * 3) % 25:02d}",
            "rating": float(3 + position % 3),
            "timestamp": 100 + position,
        }
        for position in range(60)
    ]
    pq.write_table(pa.Table.from_pylist(interactions), tmp_path / "fit.parquet")
    pq.write_table(pa.Table.from_pylist(targets), tmp_path / "test.parquet")
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t1": CUTOFF},
                "artifact_paths": {
                    "fit_interactions": "fit.parquet",
                    "test_targets": "test.parquet",
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_batch_rankings_feed_the_evaluator(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    parameters = BaselineParameters(epochs=3, embedding_dimensions=8, max_iter=20)
    outputs = {}
    for model_id in BASELINE_MODEL_IDS:
        output = tmp_path / f"{model_id}.jsonl"
        run_manifest = run_rankings_batch(
            manifest,
            model_id=model_id,
            cutoff_name="t1",
            output_path=output,
            parameters=parameters,
            k=10,
        )
        outputs[model_id] = output

        assert run_manifest["status"] == "complete"
        assert run_manifest["users"] > 0
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert all(len(row["ranked_item_ids"]) <= 10 for row in rows)
        assert "candidate_item_ids" not in rows[0]

    result = run_evaluation(
        manifest,
        model_inputs=outputs,
        output_path=tmp_path / "metrics.json",
        manifest_path=tmp_path / "metrics.manifest.json",
        run_id="batch-fixture",
        cutoff_name="t1",
        bootstrap_samples=20,
        reference_model="b0_most_popular",
    )

    assert set(result["models"]) == set(BASELINE_MODEL_IDS)
    assert result["evaluation"]["eligible_users"] == run_manifest["users"]


def test_batch_rankings_are_identical_with_workers(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    parameters = BaselineParameters(max_iter=20, ppr_batch_size=3)
    single = tmp_path / "single.jsonl"
    forked = tmp_path / "forked.jsonl"
    run_rankings_batch(
        manifest, model_id="b3_ppr", cutoff_name="t1", output_path=single, parameters=parameters
    )
    run_rankings_batch(
        manifest,
        model_id="b3_ppr",
        cutoff_name="t1",
        output_path=forked,
        parameters=parameters,
        workers=2,
    )

    def strip_timing(path: Path) -> list[dict[str, object]]:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop("latency_ms")
            row.pop("fit_time_ms")
        return rows

    assert strip_timing(single) == strip_timing(forked)


def test_batch_rankings_refuse_to_replace_a_completed_manifest(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    output = tmp_path / "b0.jsonl"
    run_rankings_batch(manifest, model_id="b0_most_popular", cutoff_name="t1", output_path=output)

    with pytest.raises(FileExistsError, match="completed"):
        run_rankings_batch(
            manifest, model_id="b0_most_popular", cutoff_name="t1", output_path=output
        )


def test_batch_rankings_save_and_reuse_a_model(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    parameters = BaselineParameters(epochs=3, embedding_dimensions=8)
    model = tmp_path / "bpr.npz"
    fitted = run_rankings_batch(
        manifest,
        model_id="b2_bpr_mf",
        cutoff_name="t1",
        output_path=tmp_path / "fitted.jsonl",
        parameters=parameters,
        save_model_path=model,
    )
    loaded = run_rankings_batch(
        manifest,
        model_id="b2_bpr_mf",
        cutoff_name="t1",
        output_path=tmp_path / "loaded.jsonl",
        parameters=parameters,
        load_model_path=model,
    )

    assert fitted["model_hash"] == loaded["model_hash"]
    assert loaded["model_source"] == "loaded"
    assert fitted["model_file"]["sha256"] == loaded["model_file"]["sha256"]

    def ranked(path: Path) -> list[list[str]]:
        return [json.loads(line)["ranked_item_ids"] for line in path.read_text().splitlines()]

    assert ranked(tmp_path / "fitted.jsonl") == ranked(tmp_path / "loaded.jsonl")
    with pytest.raises(ValueError, match="parameters"):
        run_rankings_batch(
            manifest,
            model_id="b2_bpr_mf",
            cutoff_name="t1",
            output_path=tmp_path / "other.jsonl",
            parameters=BaselineParameters(epochs=4, embedding_dimensions=8),
            load_model_path=model,
        )
