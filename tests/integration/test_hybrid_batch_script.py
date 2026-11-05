import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.run_evaluation import run_evaluation
from scripts.run_hybrid_batch import run_hybrids, tune_hybrids

from trustrec.recommenders.hybrid_batch import HybridConfig

CUTOFF = "1970-01-01T00:00:00.100000+00:00"


def _write_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    rows, evidence = [], []
    for i in range(300):
        row = {
            "review_id": f"r{i}",
            "user_id": f"u{i % 40:02d}",
            "item_id": f"i{(i * 7) % 23:02d}",
            "rating": float(1 + i % 5),
            "timestamp": i % 90,
        }
        rows.append(row)
        if i % 3:
            evidence.append(
                {
                    "review_id": row["review_id"],
                    "item_id": row["item_id"],
                    "author_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "aspect": ("story", "graphics", "value")[i % 3],
                    "sentiment": ("positive", "negative")[i % 2],
                    "confidence": 1.0,
                }
            )
    targets = [
        {
            "review_id": f"t{i}",
            "user_id": f"u{i % 45:02d}",
            "item_id": f"i{(i * 3) % 23:02d}",
            "rating": 5.0,
            "timestamp": 100 + i,
        }
        for i in range(60)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(targets), tmp_path / "targets.parquet")
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text("\n".join(json.dumps(row) for row in evidence) + "\n")
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": CUTOFF, "t1": CUTOFF},
                "artifact_paths": {
                    "train_interactions": "train.parquet",
                    "fit_interactions": "train.parquet",
                    "validation_targets": "targets.parquet",
                    "test_targets": "targets.parquet",
                },
            }
        )
    )
    return manifest, evidence_path


def test_tune_and_run_hybrids_feed_the_evaluator(tmp_path: Path) -> None:
    manifest, evidence = _write_snapshot(tmp_path)
    components = {
        "mf_parameters": {"epochs": 3, "embedding_dimensions": 4},
        "graph_parameters": {"max_iter": 10},
    }
    grid = [
        HybridConfig("h0_fixed_hybrid"),
        HybridConfig("t0_trustrec", g_max=0.25),
        HybridConfig("t0_trustrec", g_max=0.75),
    ]
    tuning = tune_hybrids(
        manifest,
        evidence_path=evidence,
        output_path=tmp_path / "tuning.json",
        grid=grid,
        bootstrap_samples=20,
        **components,
    )
    assert tuning["tuning_budget"] == {"h0_fixed_hybrid": 1, "t0_trustrec": 2}
    assert tuning["aspect_summary"]["evidence_rows"] == 200

    best_t0 = HybridConfig("t0_trustrec", **tuning["best"]["t0_trustrec"]["parameters"])
    outputs = [tmp_path / "h0.jsonl", tmp_path / "t0.jsonl"]
    manifests = run_hybrids(
        manifest,
        cutoff_name="t1",
        evidence_path=evidence,
        configs=[HybridConfig("h0_fixed_hybrid"), best_t0],
        output_paths=outputs,
        k=10,
        workers=2,
        **components,
    )
    assert [m["model_id"] for m in manifests] == ["h0_fixed_hybrid", "t0_trustrec"]
    assert manifests[1]["mean_gate"] > 0.0
    result = run_evaluation(
        manifest,
        model_inputs={"h0_fixed_hybrid": outputs[0], "t0_trustrec": outputs[1]},
        output_path=tmp_path / "metrics.json",
        manifest_path=tmp_path / "metrics.manifest.json",
        run_id="hybrid-fixture",
        cutoff_name="t1",
        bootstrap_samples=20,
    )
    assert set(result["models"]) == {"h0_fixed_hybrid", "t0_trustrec"}


def test_adr0017_settings_record_and_apply_the_recency_term(tmp_path: Path) -> None:
    from scripts.run_hybrid_batch import RECENCY_GRID

    manifest, evidence = _write_snapshot(tmp_path)
    settings = {
        "normalization": "minmax",
        "cold_start": "fallback",
        "cold_window_days": 1.0,
        "recency_window_days": 1.0,
    }
    components = {
        "mf_parameters": {"epochs": 3, "embedding_dimensions": 4},
        "graph_parameters": {"max_iter": 10},
        "scorer_settings": settings,
    }
    tuning = tune_hybrids(
        manifest,
        evidence_path=evidence,
        output_path=tmp_path / "tuning.json",
        grid=[RECENCY_GRID[0], RECENCY_GRID[-1]],
        bootstrap_samples=20,
        **components,
    )

    assert tuning["scorer_settings"] == settings
    assert tuning["best"]["t0_trustrec"]["parameters"]["tau"] == 0.8
    assert len(tuning["grid"]) == 2
    best = HybridConfig("t0_trustrec", **tuning["best"]["t0_trustrec"]["parameters"])
    manifests = run_hybrids(
        manifest,
        cutoff_name="t1",
        evidence_path=evidence,
        configs=[best],
        output_paths=[tmp_path / "t0.jsonl"],
        k=10,
        **components,
    )
    assert manifests[0]["scorer_settings"] == settings
    assert manifests[0]["parameters"]["tau"] == 0.8
