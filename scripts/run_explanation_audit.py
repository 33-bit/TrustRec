"""Run deterministic explanation claim and evidence-removal audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from trustrec.explanations.faithfulness import audit_faithfulness
from trustrec.recommenders.contracts import stable_model_hash


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _load_claims(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        recommendations = value
    elif isinstance(value, Mapping) and isinstance(value.get("recommendations"), list):
        recommendations = value["recommendations"]
    elif isinstance(value, Mapping) and isinstance(value.get("audits"), list):
        recommendations = value["audits"]
    else:
        raise ValueError("claims artifact must contain a recommendations list")
    if not all(isinstance(row, Mapping) for row in recommendations):
        raise ValueError("claims artifact recommendations must contain objects")
    return [dict(row) for row in recommendations]


def _required(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recommendation is missing {field}")
    return value


def run_audit(
    evidence_path: Path,
    claims_path: Path,
    *,
    snapshot_id: str,
    dataset_hash: str,
    cutoff_timestamp: Any,
    output_path: Path,
    seed: int = 7,
    shrinkage_lambda: float = 5.0,
    recency_half_life_days: float = 0.0,
    duplicate_penalty: str | float = "inverse_group_size",
) -> dict[str, Any]:
    """Audit prepared explanation records and write a lineage-bearing artifact."""

    snapshot_id = _required(snapshot_id, "snapshot_id")
    dataset_hash = _required(dataset_hash, "dataset_hash")
    if cutoff_timestamp in (None, ""):
        raise ValueError("cutoff_timestamp must be supplied")
    cutoff_for_artifact = (
        cutoff_timestamp.isoformat() if isinstance(cutoff_timestamp, datetime) else cutoff_timestamp
    )
    evidence_rows = _load_jsonl(evidence_path)
    recommendations = _load_claims(claims_path)
    audits: list[dict[str, Any]] = []
    for index, recommendation in enumerate(recommendations, start=1):
        recommendation_id = str(recommendation.get("recommendation_id", f"recommendation-{index}"))
        user_id = _required(recommendation.get("user_id"), "user_id")
        item_id = _required(recommendation.get("item_id"), "item_id")
        record_snapshot = recommendation.get("snapshot_id", snapshot_id)
        if record_snapshot != snapshot_id:
            raise ValueError("recommendation snapshot_id does not match audit snapshot")
        explanation = recommendation.get("explanation", recommendation)
        score_parts = recommendation.get("score_parts", recommendation.get("component_scores"))
        if not isinstance(score_parts, Mapping):
            raise ValueError("recommendation is missing score_parts")
        user_weights = recommendation.get("user_aspect_weights")
        if user_weights is not None and not isinstance(user_weights, Mapping):
            raise ValueError("user_aspect_weights must be a mapping")
        prior_scores = recommendation.get("prior_scores")
        if prior_scores is not None and not isinstance(prior_scores, Mapping):
            raise ValueError("prior_scores must be a mapping")
        audit = audit_faithfulness(
            recommendation_id=recommendation_id,
            user_id=user_id,
            item_id=item_id,
            snapshot_id=snapshot_id,
            explanation=explanation,
            evidence_rows=evidence_rows,
            cutoff_timestamp=cutoff_timestamp,
            score_parts=score_parts,
            user_aspect_weights=user_weights,
            prior_scores=prior_scores,
            shrinkage_lambda=shrinkage_lambda,
            recency_half_life_days=recency_half_life_days,
            duplicate_penalty=duplicate_penalty,
            candidate_item_ids=recommendation.get("candidate_item_ids"),
            seed=seed,
        )
        audits.append(audit.to_dict())

    configuration = {
        "seed": seed,
        "shrinkage_lambda": shrinkage_lambda,
        "recency_half_life_days": recency_half_life_days,
        "duplicate_penalty": duplicate_penalty,
        "normalization": "candidate_set_fixed",
    }
    configuration_hash = stable_model_hash(configuration)
    model_hash = stable_model_hash(
        {
            "snapshot_id": snapshot_id,
            "dataset_hash": dataset_hash,
            "cutoff_timestamp": cutoff_for_artifact,
            "configuration_hash": configuration_hash,
            "source_evidence_sha256": _sha256_file(evidence_path),
            "source_claims_sha256": _sha256_file(claims_path),
        }
    )
    artifact = {
        "schema_version": 1,
        "run_id": stable_model_hash({"model_hash": model_hash})[:16],
        "snapshot_id": snapshot_id,
        "dataset_hash": dataset_hash,
        "cutoff_timestamp": cutoff_for_artifact,
        "source_evidence_sha256": _sha256_file(evidence_path),
        "source_claims_sha256": _sha256_file(claims_path),
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "model_hash": model_hash,
        "status": "complete",
        "row_counts": {"recommendations": len(recommendations), "audits": len(audits)},
        "audits": audits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--cutoff-timestamp", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shrinkage-lambda", type=float, default=5.0)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    parser.add_argument("--duplicate-penalty", default="inverse_group_size")
    args = parser.parse_args()
    artifact = run_audit(
        args.evidence,
        args.claims,
        snapshot_id=args.snapshot_id,
        dataset_hash=args.dataset_hash,
        cutoff_timestamp=args.cutoff_timestamp,
        output_path=args.output,
        seed=args.seed,
        shrinkage_lambda=args.shrinkage_lambda,
        recency_half_life_days=args.recency_half_life_days,
        duplicate_penalty=args.duplicate_penalty,
    )
    print(json.dumps({"run_id": artifact["run_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
