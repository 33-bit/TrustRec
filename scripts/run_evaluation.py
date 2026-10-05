"""Run temporal ranking evaluation on prepared per-user ranking artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trustrec.evaluation.protocol import (
    build_evaluation_cases,
    compare_model_rankings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "evaluation-v1"
INTERACTIONS_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}
TARGETS_BY_CUTOFF = {"t0": "validation_targets", "t1": "test_targets"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _resolve_path(value: str | Path, *, base_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (base_path.parent / path, PROJECT_ROOT / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_artifact(manifest: Mapping[str, Any], manifest_path: Path, name: str) -> Path:
    value = manifest.get("artifact_paths", {}).get(name)
    if not value:
        raise ValueError(f"snapshot manifest is missing artifact_paths.{name}")
    return _resolve_path(str(value), base_path=manifest_path)


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Parquet artifact does not exist: {path}")
    return pq.read_table(path).to_pylist()


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"ranking artifact does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _load_item_aspects(path: Path | None) -> dict[str, set[str]] | None:
    if path is None:
        return None
    value = _load_json(path)
    if isinstance(value.get("item_aspects"), Mapping):
        value = value["item_aspects"]
    elif isinstance(value.get("aspects"), Mapping):
        value = value["aspects"]
    result: dict[str, set[str]] = {}
    for item_id, aspects in value.items():
        if not isinstance(item_id, str) or not isinstance(aspects, (list, tuple, set)):
            raise ValueError("item aspects must map item IDs to lists")
        result[item_id] = {str(aspect) for aspect in aspects}
    return result


def _parse_model_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use MODEL_ID=PATH")
    model_id, path = value.split("=", 1)
    if not model_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError("model must use MODEL_ID=PATH")
    return model_id.strip(), Path(path.strip())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_completed_manifest_is_absent(path: Path) -> None:
    if not path.is_file():
        return
    try:
        value = _load_json(path)
    except (ValueError, json.JSONDecodeError):
        return
    if value.get("status") == "complete":
        raise FileExistsError(f"completed evaluation manifest already exists: {path}")


def _verify_source_hash(manifest: Mapping[str, Any], name: str, path: Path) -> str:
    actual = _sha256_file(path)
    expected = manifest.get("artifact_sha256", {}).get(name)
    if expected and actual != expected:
        raise ValueError(f"{name} hash does not match the snapshot manifest")
    return actual


def run_evaluation(
    snapshot_manifest_path: Path,
    *,
    model_inputs: Mapping[str, Path],
    output_path: Path,
    manifest_path: Path,
    run_id: str,
    cutoff_name: str,
    target_path: Path | None = None,
    config_path: Path = Path("configs/evaluation.toml"),
    k_values: Sequence[int] = (5, 10, 20),
    positive_threshold: float = 4.0,
    bootstrap_seed: int = 7,
    seeds: Sequence[int] = (7, 17, 27),
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    item_aspects_path: Path | None = None,
    reference_model: str | None = None,
) -> dict[str, Any]:
    """Evaluate prepared ranking rows and write metrics and a run manifest."""

    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if cutoff_name not in INTERACTIONS_BY_CUTOFF:
        raise ValueError(f"unsupported cutoff_name: {cutoff_name}")
    if not model_inputs:
        raise ValueError("at least one model input is required")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must contain at least one integer")
    _assert_completed_manifest_is_absent(manifest_path)
    snapshot_manifest = _load_json(snapshot_manifest_path)
    for field in ("snapshot_id", "dataset_hash", "cutoffs", "artifact_paths"):
        if not snapshot_manifest.get(field):
            raise ValueError(f"snapshot manifest is missing {field}")
    cutoff_timestamp = snapshot_manifest["cutoffs"].get(cutoff_name)
    if not cutoff_timestamp:
        raise ValueError(f"snapshot manifest is missing cutoffs.{cutoff_name}")

    interaction_name = INTERACTIONS_BY_CUTOFF[cutoff_name]
    target_name = TARGETS_BY_CUTOFF[cutoff_name]
    interaction_path = _resolve_artifact(
        snapshot_manifest, snapshot_manifest_path, interaction_name
    )
    target_artifact_path = (
        _resolve_path(target_path, base_path=snapshot_manifest_path)
        if target_path is not None
        else _resolve_artifact(snapshot_manifest, snapshot_manifest_path, target_name)
    )
    source_hashes = {
        interaction_name: _verify_source_hash(
            snapshot_manifest, interaction_name, interaction_path
        ),
        target_name: _verify_source_hash(snapshot_manifest, target_name, target_artifact_path),
    }
    interactions = _load_parquet_rows(interaction_path)
    targets = _load_parquet_rows(target_artifact_path)
    dataset = build_evaluation_cases(
        interactions,
        targets,
        snapshot_id=str(snapshot_manifest["snapshot_id"]),
        cutoff_timestamp=cutoff_timestamp,
        positive_threshold=positive_threshold,
    )
    ranking_rows = {
        model_id: _load_jsonl_rows(_resolve_path(path, base_path=PROJECT_ROOT / "scripts"))
        for model_id, path in model_inputs.items()
    }
    ranking_hashes = {
        model_id: _sha256_file(_resolve_path(path, base_path=PROJECT_ROOT / "scripts"))
        for model_id, path in model_inputs.items()
    }
    item_aspects = _load_item_aspects(
        _resolve_path(item_aspects_path, base_path=snapshot_manifest_path)
        if item_aspects_path is not None
        else None
    )
    evaluations = compare_model_rankings(
        dataset,
        ranking_rows,
        k_values=k_values,
        item_aspects=item_aspects,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        reference_model=reference_model,
    )
    model_payloads: dict[str, dict[str, Any]] = {}
    for model_id, evaluation in evaluations.items():
        payload = evaluation.to_dict()
        payload["metadata"].update(
            {
                "ranking_artifact_path": str(
                    _resolve_path(model_inputs[model_id], base_path=PROJECT_ROOT / "scripts")
                ),
                "ranking_artifact_sha256": ranking_hashes[model_id],
            }
        )
        model_payloads[model_id] = payload

    config_resolved = _resolve_path(config_path, base_path=PROJECT_ROOT)
    config_hash = _sha256_file(config_resolved) if config_resolved.is_file() else None
    metrics_artifact: dict[str, Any] = {
        "schema_version": 1,
        "task": "T5.1",
        "status": "complete",
        "run_id": run_id,
        "protocol_version": PROTOCOL_VERSION,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "dataset_hash": snapshot_manifest["dataset_hash"],
        "cutoffs": snapshot_manifest["cutoffs"],
        "cutoff_name": cutoff_name,
        "cutoff_timestamp": cutoff_timestamp,
        "candidate_rule": "known_items_before_cutoff_minus_user_history",
        "target_rule": "first_new_item_with_rating_at_least_4",
        "evaluation": {
            "candidate_set_hash": dataset.candidate_set_hash,
            "eligible_users": len(dataset.cases),
            "exclusions": dict(dataset.exclusions),
            "popularity_counts": dict(dataset.popularity_counts),
        },
        "bootstrap": {
            "method": "paired_user_bootstrap",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "reference_model": reference_model,
        },
        "model_seeds": list(seeds),
        "source_artifacts": {
            name: {"path": str(path), "sha256": source_hashes[name]}
            for name, path in (
                (interaction_name, interaction_path),
                (target_name, target_artifact_path),
            )
        },
        "ranking_artifacts": {
            model_id: {
                "path": str(_resolve_path(path, base_path=PROJECT_ROOT / "scripts")),
                "sha256": ranking_hashes[model_id],
            }
            for model_id, path in model_inputs.items()
        },
        "models": model_payloads,
    }
    _write_json(output_path, metrics_artifact)
    metrics_hash = _sha256_file(output_path)
    run_manifest = {
        "schema_version": 1,
        "task": "T5.1",
        "run_id": run_id,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "protocol_version": PROTOCOL_VERSION,
        "dataset_hash": snapshot_manifest["dataset_hash"],
        "config_path": str(config_resolved),
        "config_sha256": config_hash,
        "model_ids": list(model_inputs),
        "seed": bootstrap_seed,
        "seeds": list(seeds),
        "cutoffs": snapshot_manifest["cutoffs"],
        "cutoff_name": cutoff_name,
        "metrics_path": str(output_path),
        "artifact_sha256": {
            "metrics": metrics_hash,
            **{f"source_{name}": value for name, value in source_hashes.items()},
            **{f"ranking_{model_id}": value for model_id, value in ranking_hashes.items()},
        },
        "status": "complete",
    }
    _write_json(manifest_path, run_manifest)
    return metrics_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", type=_parse_model_assignment, required=True)
    parser.add_argument("--cutoff", choices=tuple(INTERACTIONS_BY_CUTOFF), default="t1")
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.toml"))
    parser.add_argument("--k", action="append", type=int)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--seed", action="append", dest="seeds", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--reference-model")
    parser.add_argument("--item-aspects", type=Path)
    args = parser.parse_args()
    model_inputs: dict[str, Path] = {}
    for model_id, path in args.model:
        if model_id in model_inputs:
            raise SystemExit(f"duplicate model ID: {model_id}")
        model_inputs[model_id] = path
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    artifact = run_evaluation(
        args.snapshot_manifest,
        model_inputs=model_inputs,
        output_path=args.output,
        manifest_path=manifest_path,
        run_id=args.run_id,
        cutoff_name=args.cutoff,
        target_path=args.targets,
        config_path=args.config,
        k_values=tuple(args.k) if args.k else (5, 10, 20),
        positive_threshold=args.positive_threshold,
        bootstrap_seed=args.bootstrap_seed,
        seeds=tuple(args.seeds) if args.seeds else (7, 17, 27),
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        item_aspects_path=args.item_aspects,
        reference_model=args.reference_model,
    )
    print(
        json.dumps(
            {
                "run_id": artifact["run_id"],
                "status": artifact["status"],
                "models": list(artifact["models"]),
                "output": str(args.output),
                "manifest": str(manifest_path),
            }
        )
    )


if __name__ == "__main__":
    main()
