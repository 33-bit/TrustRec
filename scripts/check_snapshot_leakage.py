"""Run time and target isolation checks against a snapshot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from trustrec.evaluation.leakage import (
    assert_metadata_before,
    assert_pseudo_test_not_used_for_tuning,
    assert_review_ids_disjoint,
    assert_same_evaluation_sets,
    assert_timestamps_before,
    parse_cutoff_ms,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return parquet.read_table(path).to_pylist()


def resolve_artifact(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent.parent.parent / path


def validate_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest["artifact_paths"]
    t0 = parse_cutoff_ms(manifest["cutoffs"]["t0"])
    t1 = parse_cutoff_ms(manifest["cutoffs"]["t1"])
    if t0 >= t1:
        raise ValueError("snapshot manifest requires t0 < t1")
    train = read_rows(resolve_artifact(manifest_path, paths["train_interactions"]))
    fit = read_rows(resolve_artifact(manifest_path, paths["fit_interactions"]))
    validation_targets = read_rows(resolve_artifact(manifest_path, paths["validation_targets"]))
    test_targets = read_rows(resolve_artifact(manifest_path, paths["test_targets"]))
    assert_timestamps_before(train, t0, "train_interactions")
    assert_timestamps_before(fit, t1, "fit_interactions")
    assert_review_ids_disjoint(train, validation_targets, "validation")
    assert_review_ids_disjoint(fit, test_targets, "test")
    assert_review_ids_disjoint(train, test_targets, "train/test")
    if "item_metadata" in paths:
        metadata = read_rows(resolve_artifact(manifest_path, paths["item_metadata"]))
        if manifest.get("metadata_usage") == "ranking_feature":
            assert_metadata_before(metadata, t0, "item_metadata")
    for key in ("features", "aspect_evidence", "evidence"):
        if key in paths:
            feature_rows = read_rows(resolve_artifact(manifest_path, paths[key]))
            feature_cutoff = parse_cutoff_ms(
                manifest.get("feature_cutoffs", {}).get(key, manifest["cutoffs"]["t0"])
            )
            assert_timestamps_before(feature_rows, feature_cutoff, key)
            assert_review_ids_disjoint(feature_rows, validation_targets, key)
            assert_review_ids_disjoint(feature_rows, test_targets, key)
    if "model_evaluations" in manifest:
        assert_same_evaluation_sets(manifest["model_evaluations"])
    if "annotation_manifest" in manifest:
        assert_pseudo_test_not_used_for_tuning(manifest["annotation_manifest"])
    print(f"Leakage checks passed for {manifest['snapshot_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    validate_manifest(args.manifest)


if __name__ == "__main__":
    main()
