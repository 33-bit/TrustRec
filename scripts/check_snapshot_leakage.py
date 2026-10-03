"""Run T1.3 leakage assertions against a generated snapshot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from trustrec.evaluation.leakage import (
    assert_review_ids_disjoint,
    assert_timestamps_before,
    parse_cutoff_ms,
)


def read_rows(path: str) -> list[dict[str, Any]]:
    return parquet.read_table(path).to_pylist()


def validate_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    paths = manifest["artifact_paths"]
    t0 = parse_cutoff_ms(manifest["cutoffs"]["t0"])
    t1 = parse_cutoff_ms(manifest["cutoffs"]["t1"])
    train = read_rows(paths["train_interactions"])
    fit = read_rows(paths["fit_interactions"])
    validation_targets = read_rows(paths["validation_targets"])
    test_targets = read_rows(paths["test_targets"])
    assert_timestamps_before(train, t0, "train_interactions")
    assert_timestamps_before(fit, t1, "fit_interactions")
    assert_review_ids_disjoint(train, validation_targets, "validation")
    assert_review_ids_disjoint(fit, test_targets, "test")
    assert_review_ids_disjoint(train, test_targets, "train/test")
    print(f"Leakage checks passed for {manifest['snapshot_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    validate_manifest(args.manifest)


if __name__ == "__main__":
    main()
