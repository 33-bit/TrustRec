import json
from pathlib import Path

import pytest
from scripts.run_hybrid import run_hybrid


def _component(path: Path, model_id: str, component: str) -> None:
    path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "source_cutoff_timestamp": "2020-01-01T00:00:00+00:00",
                "model_hash": model_id,
                "result": {
                    "user_id": "u1",
                    "candidate_item_ids": ["a", "b"],
                    "items": [
                        {"item_id": "a", "total_score": 0.1, "component_scores": {component: 0.1}},
                        {"item_id": "b", "total_score": 0.8, "component_scores": {component: 0.8}},
                    ],
                },
            }
        )
    )


def test_run_hybrid_writes_common_normalization_and_lineage(tmp_path: Path) -> None:
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": "2020-01-01T00:00:00+00:00"},
            }
        )
    )
    mf = tmp_path / "mf.json"
    graph = tmp_path / "graph.json"
    _component(mf, "b2_bpr_mf", "mf")
    _component(graph, "b3_ppr", "ppr")
    aspects = tmp_path / "aspect.json"
    aspects.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoff_timestamp": "2020-01-01T00:00:00+00:00",
                "scores": {"a": 0.2, "b": 0.7},
            }
        )
    )
    support = tmp_path / "support.json"
    support.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoff_timestamp": "2020-01-01T00:00:00+00:00",
                "scores": {"a": 1.0, "b": 0.0},
            }
        )
    )
    output = tmp_path / "hybrid.json"

    artifact = run_hybrid(
        manifest,
        model_id="t0_trustrec",
        mf_result_path=mf,
        graph_result_path=graph,
        aspect_scores_path=aspects,
        aspect_support_path=support,
        output_path=output,
        history_count=0,
    )

    assert artifact["status"] == "complete"
    assert artifact["configuration"]["normalization"].startswith("percentile")
    assert artifact["source_model_hashes"]["mf"] == "b2_bpr_mf"
    assert artifact["result"]["items"][0]["item_id"] == "b"
    assert artifact["result"]["items"][0]["component_scores"]["gate"] == 0.0
    assert json.loads(output.read_text()) == artifact


def test_run_hybrid_rejects_component_cutoff_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": "2020-01-01T00:00:00+00:00"},
            }
        )
    )
    mf = tmp_path / "mf.json"
    graph = tmp_path / "graph.json"
    _component(mf, "b2_bpr_mf", "mf")
    _component(graph, "b3_ppr", "ppr")
    graph_payload = json.loads(graph.read_text())
    graph_payload["source_cutoff_timestamp"] = "2020-01-02T00:00:00+00:00"
    graph.write_text(json.dumps(graph_payload))
    aspects = tmp_path / "aspect.json"
    aspects.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoff_timestamp": "2020-01-01T00:00:00+00:00",
                "scores": {"a": 0.2, "b": 0.7},
            }
        )
    )
    with pytest.raises(ValueError, match="cutoff"):
        run_hybrid(
            manifest,
            model_id="h0_fixed_hybrid",
            mf_result_path=mf,
            graph_result_path=graph,
            aspect_scores_path=aspects,
            output_path=tmp_path / "hybrid.json",
        )
