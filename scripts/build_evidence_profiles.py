"""Build snapshot-scoped item-aspect evidence profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from trustrec.explanations.evidence import aggregate_evidence
from trustrec.recommenders.contracts import stable_model_hash


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_profiles(
    evidence_path: Path,
    *,
    snapshot_id: str,
    dataset_hash: str,
    cutoff_timestamp: str,
    output_path: Path,
    prior_scores: dict[str, float] | None = None,
    shrinkage_lambda: float = 5.0,
    recency_half_life_days: float = 0.0,
    duplicate_penalty: str | float = "inverse_group_size",
    items: list[str] | None = None,
    aspects: list[str] | None = None,
) -> dict[str, Any]:
    """Read JSON Lines evidence and write a lineage-bearing profile artifact."""

    if not snapshot_id.strip() or not dataset_hash.strip():
        raise ValueError("snapshot_id and dataset_hash must be non-empty")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"evidence artifact does not exist: {evidence_path}")
    rows: list[dict[str, Any]] = []
    with evidence_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{evidence_path}:{line_number} must contain an object")
            rows.append(value)
    profiles = aggregate_evidence(
        rows,
        cutoff_timestamp=cutoff_timestamp,
        prior_scores=prior_scores,
        shrinkage_lambda=shrinkage_lambda,
        recency_half_life_days=recency_half_life_days,
        duplicate_penalty=duplicate_penalty,
        items=items,
        aspects=aspects,
    )
    configuration = {
        "shrinkage_lambda": shrinkage_lambda,
        "recency_half_life_days": recency_half_life_days,
        "duplicate_penalty": duplicate_penalty,
        "items": items,
        "aspects": aspects,
    }
    configuration_hash = stable_model_hash(configuration)
    artifact = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "dataset_hash": dataset_hash,
        "source_evidence_sha256": _sha256_file(evidence_path),
        "cutoff_timestamp": cutoff_timestamp,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "model_hash": stable_model_hash(
            {
                "snapshot_id": snapshot_id,
                "dataset_hash": dataset_hash,
                "cutoff_timestamp": cutoff_timestamp,
                "configuration_hash": configuration_hash,
                "profiles": {
                    f"{item_id}:{aspect}": profile.to_dict()
                    for (item_id, aspect), profile in sorted(profiles.items())
                },
            }
        ),
        "status": "complete",
        "profiles": [
            profile.to_dict() for _, profile in sorted(profiles.items(), key=lambda pair: pair[0])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--cutoff-timestamp", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shrinkage-lambda", type=float, default=5.0)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    parser.add_argument("--items", nargs="*")
    parser.add_argument("--aspects", nargs="*")
    args = parser.parse_args()
    artifact = build_profiles(
        args.evidence,
        snapshot_id=args.snapshot_id,
        dataset_hash=args.dataset_hash,
        cutoff_timestamp=args.cutoff_timestamp,
        output_path=args.output,
        shrinkage_lambda=args.shrinkage_lambda,
        recency_half_life_days=args.recency_half_life_days,
        items=args.items,
        aspects=args.aspects,
    )
    print(json.dumps({"snapshot_id": artifact["snapshot_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
