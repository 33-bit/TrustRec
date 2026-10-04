"""Run one T3.1 ranking baseline from a snapshot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trustrec.recommenders.contracts import (
    build_candidate_set,
    stable_model_hash,
)
from trustrec.recommenders.item_knn import rank_item_knn
from trustrec.recommenders.popularity import rank_popularity

MODEL_IDS = ("b0_most_popular", "b1_item_knn")
ARTIFACT_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"snapshot manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for field in ("snapshot_id", "dataset_hash", "cutoffs", "artifact_paths"):
        if not manifest.get(field):
            raise ValueError(f"snapshot manifest is missing {field}")
    return manifest


def _resolve_artifact(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (
        manifest_path.resolve().parent / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return PROJECT_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def run_baseline(
    manifest_path: Path,
    *,
    model_id: str,
    user_id: str,
    cutoff_name: str,
    output_path: Path,
    positive_threshold: float = 4.0,
    neighbor_limit: int = 0,
    seed: int | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Fit and write one baseline result from a prepared snapshot."""

    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported model_id: {model_id}")
    if cutoff_name not in ARTIFACT_BY_CUTOFF:
        raise ValueError(f"unsupported cutoff_name: {cutoff_name}")
    if neighbor_limit < 0:
        raise ValueError("neighbor_limit cannot be negative")
    manifest = _load_manifest(manifest_path)
    cutoff_timestamp = manifest["cutoffs"].get(cutoff_name)
    if not cutoff_timestamp:
        raise ValueError(f"snapshot manifest is missing cutoffs.{cutoff_name}")
    artifact_name = ARTIFACT_BY_CUTOFF[cutoff_name]
    artifact_value = manifest["artifact_paths"].get(artifact_name)
    if not artifact_value:
        raise ValueError(f"snapshot manifest is missing artifact_paths.{artifact_name}")
    interactions_path = _resolve_artifact(manifest_path, artifact_value)
    if not interactions_path.is_file():
        raise FileNotFoundError(f"interaction artifact does not exist: {interactions_path}")
    expected_artifact_hash = manifest.get("artifact_sha256", {}).get(artifact_name)
    source_artifact_hash = _sha256_file(interactions_path)
    if expected_artifact_hash and source_artifact_hash != expected_artifact_hash:
        raise ValueError(f"{artifact_name} hash does not match the snapshot manifest")
    interactions = pq.read_table(interactions_path).to_pylist()
    candidates = build_candidate_set(
        user_id=user_id,
        snapshot_id=manifest["snapshot_id"],
        interactions=interactions,
        cutoff_timestamp=cutoff_timestamp,
    )
    if model_id == "b0_most_popular":
        result = rank_popularity(
            interactions,
            candidates,
            positive_threshold=positive_threshold,
            snapshot_id=manifest["snapshot_id"],
            seed=seed,
            k=k,
        )
    else:
        result = rank_item_knn(
            interactions,
            candidates,
            positive_threshold=positive_threshold,
            n_neighbors=neighbor_limit or None,
            snapshot_id=manifest["snapshot_id"],
            seed=seed,
            k=k,
        )
    configuration_hash = stable_model_hash(
        {
            "model_id": model_id,
            "cutoff_name": cutoff_name,
            "positive_threshold": positive_threshold,
            "neighbor_limit": neighbor_limit,
            "seed": seed,
            "k": k,
        }
    )
    artifact = {
        "schema_version": 1,
        "run_id": stable_model_hash(
            {
                "snapshot_id": manifest["snapshot_id"],
                "dataset_hash": manifest["dataset_hash"],
                "model_hash": result.model_hash,
                "configuration_hash": configuration_hash,
                "user_id": user_id,
                "cutoff_name": cutoff_name,
                "seed": seed,
                "k": k,
            }
        )[:16],
        "model_id": model_id,
        "snapshot_id": manifest["snapshot_id"],
        "dataset_hash": manifest["dataset_hash"],
        "source_artifact_sha256": source_artifact_hash,
        "source_cutoff_timestamp": cutoff_timestamp,
        "cutoff_name": cutoff_name,
        "seed": seed,
        "configuration": result.configuration,
        "configuration_hash": configuration_hash,
        "model_hash": result.model_hash,
        "status": "complete",
        "result": result.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--cutoff", choices=tuple(ARTIFACT_BY_CUTOFF), default="t0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--neighbor-limit", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k", type=_positive_integer)
    args = parser.parse_args()
    artifact = run_baseline(
        args.snapshot_manifest,
        model_id=args.model,
        user_id=args.user_id,
        cutoff_name=args.cutoff,
        output_path=args.output,
        positive_threshold=args.positive_threshold,
        neighbor_limit=args.neighbor_limit,
        seed=args.seed,
        k=args.k,
    )
    print(
        json.dumps(
            {
                "run_id": artifact["run_id"],
                "model_id": artifact["model_id"],
                "snapshot_id": artifact["snapshot_id"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
