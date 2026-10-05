"""Run the fixed H0 or adaptive T0 hybrid from prepared component artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trustrec.recommenders.contracts import stable_model_hash, timestamp_value
from trustrec.recommenders.hybrid import (
    adaptive_trustrec_details,
    fixed_hybrid_details,
    rank_scores,
)

MODEL_IDS = ("h0_fixed_hybrid", "t0_trustrec")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_result(artifact: Mapping[str, Any], component: str) -> dict[str, float]:
    result = artifact.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("component artifact is missing result")
    items = result.get("items")
    if not isinstance(items, list):
        raise ValueError("component artifact result is missing items")
    scores: dict[str, float] = {}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str):
            raise ValueError("component artifact contains an invalid item")
        components = item.get("component_scores", {})
        if not isinstance(components, Mapping):
            raise ValueError("component artifact item is missing component_scores")
        value = components.get(component, item.get("total_score"))
        if value is None:
            raise ValueError(f"component artifact has no {component} score")
        scores[item["item_id"]] = float(value)
    if len(scores) != len(items):
        raise ValueError("component artifact contains duplicate item IDs")
    candidate_ids = result.get("candidate_item_ids")
    if not isinstance(candidate_ids, (list, tuple)):
        raise ValueError("component artifact must declare candidate_item_ids")
    if set(candidate_ids) != set(scores):
        raise ValueError("component artifact must score every candidate item")
    return scores


def _load_score_map(path: Path, field: str = "score") -> tuple[dict[str, float], Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    metadata: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
    if isinstance(value, Mapping):
        if isinstance(value.get("scores"), Mapping):
            value = value["scores"]
        elif isinstance(value.get(field), Mapping):
            value = value[field]
        elif isinstance(value.get("aspect_scores"), Mapping):
            value = value["aspect_scores"]
        elif isinstance(value.get("result"), Mapping):
            value = value["result"]
        if all(isinstance(key, str) for key in value):
            return {key: float(score) for key, score in value.items()}, metadata
    if isinstance(value, list):
        result: dict[str, float] = {}
        for row in value:
            if not isinstance(row, Mapping) or not isinstance(row.get("item_id"), str):
                raise ValueError(f"{field} score list contains an invalid row")
            score = row.get(field, row.get("total_score"))
            if score is None:
                raise ValueError(f"{field} score list contains a row without {field}")
            result[row["item_id"]] = float(score)
        return result, metadata
    raise ValueError(f"{field} artifact must be a mapping or list")


def _validate_lineage(
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    label: str,
    expected_cutoff: Any,
) -> None:
    for field in ("snapshot_id", "dataset_hash"):
        if artifact.get(field) != manifest.get(field):
            raise ValueError(f"{label} {field} does not match snapshot manifest")
    result = artifact.get("result")
    result_cutoff = result.get("cutoff_timestamp") if isinstance(result, Mapping) else None
    cutoffs = [
        value
        for value in (
            artifact.get("source_cutoff_timestamp"),
            artifact.get("cutoff_timestamp"),
            result_cutoff,
        )
        if value is not None
    ]
    if not cutoffs or any(
        timestamp_value(value) != timestamp_value(expected_cutoff) for value in cutoffs
    ):
        raise ValueError(f"{label} cutoff does not match snapshot manifest")


def run_hybrid(
    manifest_path: Path,
    *,
    model_id: str,
    mf_result_path: Path,
    graph_result_path: Path,
    aspect_scores_path: Path,
    output_path: Path,
    aspect_support_path: Path | None = None,
    history_count: int | None = None,
    cutoff_name: str = "t0",
    mf_weight: float = 0.5,
    graph_weight: float = 0.25,
    aspect_weight: float = 0.25,
    kappa: float = 5.0,
    g_max: float = 0.5,
    rho: float = 0.5,
    k: int | None = None,
) -> dict[str, Any]:
    """Combine prepared component artifacts and write one hybrid artifact."""

    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported model_id: {model_id}")
    history_was_supplied = history_count is not None
    if history_count is None:
        history_count = 0
    elif isinstance(history_count, bool) or not isinstance(history_count, int) or history_count < 0:
        raise ValueError("history_count must be a non-negative integer")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError("k must be a positive integer or None")
    manifest = _load_json(manifest_path)
    for field in ("snapshot_id", "dataset_hash", "cutoffs"):
        if not manifest.get(field):
            raise ValueError(f"snapshot manifest is missing {field}")
    cutoff_timestamp = manifest["cutoffs"].get(cutoff_name)
    if not cutoff_timestamp:
        raise ValueError(f"snapshot manifest is missing cutoffs.{cutoff_name}")
    mf_artifact = _load_json(mf_result_path)
    graph_artifact = _load_json(graph_result_path)
    _validate_lineage(mf_artifact, manifest, "MF artifact", cutoff_timestamp)
    _validate_lineage(graph_artifact, manifest, "graph artifact", cutoff_timestamp)
    if mf_artifact.get("model_id") != "b2_bpr_mf":
        raise ValueError("MF artifact must use model_id b2_bpr_mf")
    if graph_artifact.get("model_id") != "b3_ppr":
        raise ValueError("graph artifact must use model_id b3_ppr")
    mf_scores = _component_result(mf_artifact, "mf")
    graph_scores = _component_result(graph_artifact, "ppr")
    aspect_scores, aspect_metadata = _load_score_map(aspect_scores_path, "score")
    _validate_lineage(aspect_metadata, manifest, "aspect scores", cutoff_timestamp)
    if set(mf_scores) != set(graph_scores) or set(mf_scores) != set(aspect_scores):
        raise ValueError("all hybrid components must use the same candidate IDs")
    if aspect_support_path is None:
        aspect_support = {item_id: 0.0 for item_id in mf_scores}
        support_metadata: Mapping[str, Any] = {}
    else:
        aspect_support, support_metadata = _load_score_map(aspect_support_path, "support")
        _validate_lineage(support_metadata, manifest, "aspect support", cutoff_timestamp)
    if set(aspect_support) != set(mf_scores):
        raise ValueError("aspect support must use the same candidate IDs")

    fallback_reasons: list[str] = []
    for source_artifact in (mf_artifact, graph_artifact):
        result = source_artifact.get("result")
        if isinstance(result, Mapping) and result.get("fallback_reason"):
            fallback_reasons.append(str(result["fallback_reason"]))
    if model_id == "t0_trustrec" and aspect_support_path is None:
        fallback_reasons.append("no_aspect_support")
    if model_id == "t0_trustrec" and not history_was_supplied:
        fallback_reasons.append("no_history_count")
    configuration = {
        "model_id": model_id,
        "cutoff_name": cutoff_name,
        "history_count": history_count,
        "mf_weight": mf_weight,
        "graph_weight": graph_weight,
        "aspect_weight": aspect_weight,
        "kappa": kappa,
        "g_max": g_max,
        "rho": rho,
        "k": k,
        "normalization": "percentile_within_candidate_set_average_ties",
        "tuning_budget": "shared_candidate_validation_grid",
    }
    if model_id == "h0_fixed_hybrid":
        details = fixed_hybrid_details(
            mf_scores,
            graph_scores,
            aspect_scores,
            mf_weight=mf_weight,
            graph_weight=graph_weight,
            aspect_weight=aspect_weight,
        )
    else:
        details = adaptive_trustrec_details(
            mf_scores,
            graph_scores,
            aspect_scores,
            aspect_support,
            history_count,
            kappa=kappa,
            g_max=g_max,
            rho=rho,
        )
    ordered = rank_scores({item_id: value.total_score for item_id, value in details.items()}, k=k)
    items = [
        {
            "item_id": item_id,
            "rank": rank,
            "total_score": details[item_id].total_score,
            "component_scores": details[item_id].component_scores,
        }
        for rank, (item_id, _) in enumerate(ordered, start=1)
    ]
    configuration_hash = stable_model_hash(configuration)
    model_hash = stable_model_hash(
        {
            "model_id": model_id,
            "configuration_hash": configuration_hash,
            "mf_scores": mf_scores,
            "graph_scores": graph_scores,
            "aspect_scores": aspect_scores,
            "aspect_support": aspect_support,
        }
    )
    artifact = {
        "schema_version": 1,
        "run_id": stable_model_hash(
            {
                "snapshot_id": manifest["snapshot_id"],
                "dataset_hash": manifest["dataset_hash"],
                "model_hash": model_hash,
                "user_id": mf_artifact.get("result", {}).get("user_id"),
                "cutoff_name": cutoff_name,
            }
        )[:16],
        "model_id": model_id,
        "snapshot_id": manifest["snapshot_id"],
        "dataset_hash": manifest["dataset_hash"],
        "source_cutoff_timestamp": cutoff_timestamp,
        "source_model_hashes": {
            "mf": mf_artifact.get("model_hash"),
            "graph": graph_artifact.get("model_hash"),
            "aspect": _sha256_file(aspect_scores_path),
            "aspect_support": _sha256_file(aspect_support_path) if aspect_support_path else None,
        },
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "model_hash": model_hash,
        "status": "complete",
        "fallback_reason": ";".join(dict.fromkeys(fallback_reasons)) or None,
        "result": {
            "user_id": mf_artifact.get("result", {}).get("user_id"),
            "snapshot_id": manifest["snapshot_id"],
            "cutoff_timestamp": cutoff_timestamp,
            "candidate_item_ids": sorted(mf_scores),
            "items": items,
            "fallback_reason": ";".join(dict.fromkeys(fallback_reasons)) or None,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--mf-result", type=Path, required=True)
    parser.add_argument("--graph-result", type=Path, required=True)
    parser.add_argument("--aspect-scores", type=Path, required=True)
    parser.add_argument("--aspect-support", type=Path)
    parser.add_argument("--history-count", type=int)
    parser.add_argument("--cutoff", choices=("t0", "t1"), default="t0")
    parser.add_argument("--mf-weight", type=float, default=0.5)
    parser.add_argument("--graph-weight", type=float, default=0.25)
    parser.add_argument("--aspect-weight", type=float, default=0.25)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--g-max", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--k", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_hybrid(
        args.snapshot_manifest,
        model_id=args.model,
        mf_result_path=args.mf_result,
        graph_result_path=args.graph_result,
        aspect_scores_path=args.aspect_scores,
        aspect_support_path=args.aspect_support,
        history_count=args.history_count,
        cutoff_name=args.cutoff,
        mf_weight=args.mf_weight,
        graph_weight=args.graph_weight,
        aspect_weight=args.aspect_weight,
        kappa=args.kappa,
        g_max=args.g_max,
        rho=args.rho,
        k=args.k,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "run_id": artifact["run_id"],
                "model_id": artifact["model_id"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
