"""Validate the frozen LLM pseudo-test records and split manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

try:
    from scripts.validate_ai_pilot_labels import (
        _compare_records,
        _jsonl_record,
        _validate_record,
        _validate_source_snapshot,
    )
except ModuleNotFoundError:
    from validate_ai_pilot_labels import (  # type: ignore[no-redef]
        _compare_records,
        _jsonl_record,
        _validate_record,
        _validate_source_snapshot,
    )

ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
T2_2_CONTRACT = "t2.2-full-snapshot-v1"
T2_2_UNIT_COUNT = 100
T2_2_RATING_QUOTA = 20
REQUIRED_PROHIBITED_USES = {
    "prompt_selection",
    "threshold_tuning",
    "model_tuning",
    "hyperparameter_tuning",
    "recommendation_metrics",
    "human_gold_claim",
    "ground_truth_claim",
}
PROVENANCE_FIELDS = (
    "source_snapshot_id",
    "source_cutoff_timestamp",
    "split_role",
    "label_status",
    "model_id",
    "model_revision",
    "prompt_version",
    "temperature",
    "seed",
    "generated_at",
    "source_snapshot_dataset_hash",
    "configuration_hash",
    "model_hash",
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _resolve_source_artifact(source_manifest_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (source_manifest_path.parent.parent.parent / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str, field: str) -> None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} is not an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a time zone")


def _parse_unavailable_number(value: Any, field: str) -> Any:
    if value == "unavailable":
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field} is not a number or unavailable")
    if field == "temperature":
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} is not a number or unavailable") from error
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"{field} must be finite and non-negative")
        return parsed
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not an integer or unavailable") from error


def _hash_model(manifest: dict[str, Any]) -> str:
    payload = {
        field: manifest[field]
        for field in ("model_id", "model_revision", "prompt_version", "temperature", "seed")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_configuration(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for listed_path in manifest["configuration_files"]:
        path = _resolve(listed_path)
        if not path.is_file():
            raise ValueError(f"configuration file does not exist: {listed_path}")
        digest.update(str(listed_path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_manifest(manifest: dict[str, Any], *, enforce_t2_2_contract: bool) -> None:
    required = {
        "schema_version",
        "artifact",
        "jsonl_artifact",
        "selection_artifact",
        "unit_count",
        "source_dataset",
        "source_provider",
        "source_snapshot_id",
        "source_snapshot_dataset_hash",
        "source_cutoff_timestamp",
        "split_role",
        "label_status",
        "frozen",
        "frozen_at",
        "model_id",
        "model_revision",
        "prompt_version",
        "temperature",
        "seed",
        "generated_at",
        "source_text_hash_field",
        "usage_restrictions",
        "recommendation_test",
        "annotation_covers_entire_corpus",
        "configuration_files",
        "configuration_hash",
        "model_hash",
        "prompt_file",
        "prompt_hash",
        "pilot_artifact",
        "pilot_artifact_sha256",
        "source_snapshot_manifest",
        "hash_definition",
        "selection",
        "artifact_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("manifest lacks required fields: " + ", ".join(missing))
    if manifest["schema_version"] != 2:
        raise ValueError("manifest must use schema version 2")
    if manifest["split_role"] != "llm_pseudo_test" or manifest["label_status"] != "llm_pseudo_test":
        raise ValueError("manifest must describe llm_pseudo_test labels")
    if manifest["frozen"] is not True:
        raise ValueError("llm_pseudo_test manifest must be frozen")
    if not isinstance(manifest["unit_count"], int) or manifest["unit_count"] <= 0:
        raise ValueError("manifest unit_count must be a positive integer")
    for field in ("source_cutoff_timestamp", "generated_at", "frozen_at"):
        _parse_timestamp(manifest[field], field)
    for field in (
        "source_snapshot_dataset_hash",
        "configuration_hash",
        "model_hash",
        "prompt_hash",
    ):
        if not isinstance(manifest[field], str) or not SHA256_RE.fullmatch(manifest[field]):
            raise ValueError(f"manifest {field} must be a lowercase SHA-256 hash")
    for field in ("temperature", "seed"):
        _parse_unavailable_number(manifest[field], field)
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"]:
        raise ValueError("manifest model_id must be non-empty")
    if not isinstance(manifest["model_revision"], str) or not manifest["model_revision"]:
        raise ValueError("manifest model_revision must be explicit")
    if not SHA256_RE.fullmatch(manifest["pilot_artifact_sha256"]):
        raise ValueError("manifest pilot_artifact_sha256 is invalid")
    if manifest["model_hash"] != _hash_model(manifest):
        raise ValueError("manifest model_hash does not match the model provenance")
    if manifest["configuration_hash"] != _hash_configuration(manifest):
        raise ValueError("manifest configuration_hash does not match the configuration files")
    prompt_path = _resolve(manifest["prompt_file"])
    if not prompt_path.is_file():
        raise ValueError(f"prompt file does not exist: {manifest['prompt_file']}")
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != manifest["prompt_hash"]:
        raise ValueError("manifest prompt_hash does not match the prompt file")
    restrictions = manifest["usage_restrictions"]
    if (
        not isinstance(restrictions, dict)
        or not isinstance(restrictions.get("allowed_uses"), list)
        or not isinstance(restrictions.get("prohibited_uses"), list)
    ):
        raise ValueError("manifest usage_restrictions must contain lists")
    if any(
        term in " ".join(map(str, restrictions["allowed_uses"])).casefold()
        for term in ("tuning", "prompt", "threshold", "hyperparameter")
    ):
        raise ValueError("frozen llm_pseudo_test usage restrictions allow tuning")
    if not REQUIRED_PROHIBITED_USES <= set(map(str, restrictions["prohibited_uses"])):
        raise ValueError("frozen llm_pseudo_test prohibited uses are incomplete")
    if manifest["recommendation_test"] is not False:
        raise ValueError("llm_pseudo_test cannot be a recommendation_test")
    if manifest["annotation_covers_entire_corpus"] is not False:
        raise ValueError("llm_pseudo_test cannot cover the entire corpus")
    if manifest["source_text_hash_field"] != "source_text_sha256":
        raise ValueError("manifest source_text_hash_field must be source_text_sha256")
    selection = manifest["selection"]
    if not isinstance(selection, dict) or not isinstance(selection.get("seed"), int):
        raise ValueError("manifest selection must include an integer seed")
    if enforce_t2_2_contract and selection.get("contract") != T2_2_CONTRACT:
        raise ValueError(f"manifest selection contract must be {T2_2_CONTRACT}")
    if selection.get("contract") == T2_2_CONTRACT:
        if manifest["unit_count"] != T2_2_UNIT_COUNT:
            raise ValueError(f"{T2_2_CONTRACT} requires {T2_2_UNIT_COUNT} units")
        if selection.get("rating_quota") != T2_2_RATING_QUOTA:
            raise ValueError(f"{T2_2_CONTRACT} requires a rating quota of {T2_2_RATING_QUOTA}")
    artifacts = manifest["artifact_sha256"]
    if not isinstance(artifacts, dict) or not {"labels", "jsonl", "selection"} <= set(artifacts):
        raise ValueError("manifest artifact_sha256 is incomplete")
    for field, value in artifacts.items():
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"manifest artifact_sha256.{field} is invalid")


def _csv_record(row: dict[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = dict(row)
    for field in ("schema_version", "timestamp", "unit_start", "unit_end"):
        try:
            record[field] = int(record[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"CSV field {field} is not an integer") from error
    for field in ("temperature", "seed"):
        record[field] = _parse_unavailable_number(record[field], field)
    for field in ("labels", "evidence_offsets", "usage_restrictions"):
        csv_field = f"{field}_json"
        try:
            record[field] = json.loads(record.pop(csv_field))
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"CSV field {csv_field} is not valid JSON") from error
    for field in ("out_of_scope", "needs_review"):
        if isinstance(record[field], str) and record[field].casefold() in {"true", "false"}:
            record[field] = record[field].casefold() == "true"
            continue
        if record[field] not in {True, False}:
            raise ValueError(f"CSV field {field} is not a boolean")
    return record


def _load_source_index(
    manifest: dict[str, Any], *, require_full_snapshot: bool = False
) -> dict[str, dict[str, Any]]:
    source_manifest_path, source_manifest = _load_source_manifest(
        manifest, require_full_snapshot=require_full_snapshot
    )
    source_path = _resolve_source_artifact(
        source_manifest_path, source_manifest["artifact_paths"]["review_texts"]
    )
    source_index: dict[str, dict[str, Any]] = {}
    for batch in parquet.ParquetFile(source_path).iter_batches(
        columns=["review_id", "item_id", "text", "timestamp", "duplicate_group"],
        use_threads=False,
    ):
        for row in batch.to_pylist():
            source_index[row["review_id"]] = row
    interaction_path = _resolve_source_artifact(
        source_manifest_path, source_manifest["artifact_paths"]["interactions"]
    )
    for batch in parquet.ParquetFile(interaction_path).iter_batches(
        columns=["review_id", "rating"], use_threads=False
    ):
        for row in batch.to_pylist():
            if row["review_id"] in source_index:
                source_index[row["review_id"]]["rating"] = float(row["rating"])
    return source_index


def _load_source_manifest(
    manifest: dict[str, Any], *, require_full_snapshot: bool = False
) -> tuple[Path, dict[str, Any]]:
    source_manifest_path = _resolve(manifest["source_snapshot_manifest"])
    if not source_manifest_path.is_file():
        raise ValueError("source snapshot manifest does not exist")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("snapshot_id") != manifest["source_snapshot_id"]:
        raise ValueError("source snapshot ID does not match the pseudo-test manifest")
    if source_manifest.get("dataset_hash") != manifest["source_snapshot_dataset_hash"]:
        raise ValueError("source snapshot dataset hash does not match the pseudo-test manifest")
    source_cutoff = source_manifest.get("cutoffs", {}).get("t0")
    if require_full_snapshot and source_manifest.get("schema_version") != 3:
        raise ValueError("T2.2 requires a schema version 3 full snapshot manifest")
    if require_full_snapshot and not source_cutoff:
        raise ValueError("source snapshot manifest lacks cutoffs.t0")
    if source_cutoff and source_cutoff != manifest["source_cutoff_timestamp"]:
        raise ValueError("source snapshot cutoff does not match the pseudo-test manifest")
    if source_cutoff:
        _parse_timestamp(source_cutoff, "source snapshot cutoff")
    return source_manifest_path, source_manifest


def _validate_source_artifacts(
    manifest: dict[str, Any],
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    *,
    require_full_snapshot: bool = False,
) -> None:
    artifact_paths = source_manifest.get("artifact_paths", {})
    artifact_hashes = source_manifest.get("artifact_sha256", {})
    required_names = ("review_texts", "interactions")
    for name in required_names:
        if name not in artifact_paths or name not in artifact_hashes:
            raise ValueError(f"source manifest lacks {name} artifact hash")
    for name in artifact_paths:
        if name not in artifact_hashes:
            raise ValueError(f"source manifest lacks {name} artifact hash")
        path = _resolve_source_artifact(source_manifest_path, artifact_paths[name])
        if not path.is_file():
            raise ValueError(f"source artifact does not exist: {artifact_paths[name]}")
        expected = artifact_hashes[name]
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise ValueError(f"source artifact hash is invalid for {name}")
        if _sha256_file(path) != expected:
            raise ValueError(f"source artifact hash mismatch for {name}")
    if require_full_snapshot and source_manifest.get("schema_version") != 3:
        raise ValueError("T2.2 requires a schema version 3 full snapshot manifest")
    if source_manifest.get("schema_version") == 3:
        source_payload = {
            "artifact_sha256": source_manifest["artifact_sha256"],
            "cutoffs": {
                "t0": round(
                    datetime.fromisoformat(
                        source_manifest["cutoffs"]["t0"].replace("Z", "+00:00")
                    ).timestamp()
                    * 1000
                ),
                "t1": round(
                    datetime.fromisoformat(
                        source_manifest["cutoffs"]["t1"].replace("Z", "+00:00")
                    ).timestamp()
                    * 1000
                ),
            },
            "metadata_source_sha256": source_manifest["sources"]["metadata"]["source_sha256"],
            "review_source_sha256": source_manifest["sources"]["reviews"]["source_sha256"],
            "scope": "software_games_only_after_metadata_join",
        }
        dataset_hash = hashlib.sha256(
            json.dumps(
                source_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if source_manifest.get("dataset_hash") != dataset_hash:
            raise ValueError("source snapshot dataset hash does not match its artifacts")


def _validate_split_independence(
    records: list[dict[str, Any]], manifest: dict[str, Any], source_index: dict[str, dict[str, Any]]
) -> None:
    pilot_path = _resolve(manifest["pilot_artifact"])
    if _sha256_file(pilot_path) != manifest["pilot_artifact_sha256"]:
        raise ValueError("pilot artifact hash does not match the frozen manifest")
    pilot_records = [
        json.loads(line)
        for line in pilot_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pilot_ids = {str(row["review_id"]) for row in pilot_records}
    pseudo_ids = {str(row["review_id"]) for row in records}
    overlap = pilot_ids & pseudo_ids
    if overlap:
        raise ValueError(f"pilot review overlap: {len(overlap)} review IDs")
    pilot_groups = {
        (
            str(source_index[row["review_id"]]["item_id"]),
            str(source_index[row["review_id"]]["duplicate_group"]),
        )
        for row in pilot_records
        if row.get("review_id") in source_index
    }
    pseudo_groups = {
        (
            str(source_index[row["review_id"]]["item_id"]),
            str(source_index[row["review_id"]]["duplicate_group"]),
        )
        for row in records
    }
    if pilot_groups & pseudo_groups:
        raise ValueError(
            f"pilot duplicate-group overlap: {len(pilot_groups & pseudo_groups)} groups"
        )


def _validate_selection(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    source_index: dict[str, dict[str, Any]],
    *,
    enforce_t2_2_contract: bool,
) -> None:
    selection_path = _resolve(manifest["selection_artifact"])
    selection = [
        json.loads(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(selection) != len(records):
        raise ValueError("selection count does not match label count")
    if enforce_t2_2_contract:
        rating_counts = Counter()
        if len(records) != T2_2_UNIT_COUNT:
            raise ValueError(f"{T2_2_CONTRACT} requires {T2_2_UNIT_COUNT} units")
    else:
        rating_counts = None
    seen_review_ids: set[str] = set()
    for rank, (selected, record) in enumerate(zip(selection, records, strict=True), start=1):
        if (
            selected.get("selection_rank") != rank
            or selected.get("review_id") != record["review_id"]
        ):
            raise ValueError(f"selection order mismatch at rank {rank}")
        source = source_index.get(record["review_id"])
        if source is None:
            raise ValueError(f"source snapshot is missing review {record['review_id']}")
        if record["review_id"] in seen_review_ids:
            raise ValueError("duplicate review ID")
        seen_review_ids.add(record["review_id"])
        if (
            selected.get("item_id") != source["item_id"]
            or selected.get("duplicate_group") != source["duplicate_group"]
        ):
            raise ValueError(f"selection source ID mismatch for {record['review_id']}")
        if selected.get("source_text_sha256") != record["source_text_sha256"]:
            raise ValueError(f"selection source text hash mismatch for {record['review_id']}")
        if selected.get("selection_seed") != manifest["selection"]["seed"]:
            raise ValueError("selection seed does not match the manifest")
        if round(float(selected["rating"])) != round(float(source["rating"])):
            raise ValueError(f"selection rating mismatch for {record['review_id']}")
        if rating_counts is not None:
            rating_counts[round(float(source["rating"]))] += 1
    if rating_counts is not None:
        expected = {rating: T2_2_RATING_QUOTA for rating in range(1, 6)}
        if dict(rating_counts) != expected:
            raise ValueError(
                f"{T2_2_CONTRACT} requires 20 reviews per rating: {dict(rating_counts)}"
            )


def _validate_pseudo_test_artifacts(
    labels_path: Path,
    jsonl_path: Path,
    manifest_path: Path,
    *,
    enforce_t2_2_contract: bool,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest, enforce_t2_2_contract=enforce_t2_2_contract)
    expected_paths = {
        "artifact": labels_path,
        "jsonl_artifact": jsonl_path,
        "selection_artifact": _resolve(manifest["selection_artifact"]),
    }
    for field, path in expected_paths.items():
        listed = _resolve(manifest[field])
        if listed != path.resolve():
            raise ValueError(f"{field} path does not match the supplied artifact")
    for field, path in (
        ("labels", labels_path),
        ("jsonl", jsonl_path),
        ("selection", expected_paths["selection_artifact"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifact_sha256"][field]:
            raise ValueError(f"{field} artifact hash does not match the frozen manifest")
    with labels_path.open(encoding="utf-8", newline="") as handle:
        csv_records = [_csv_record(row) for row in csv.DictReader(handle)]
    if len(csv_records) != manifest["unit_count"]:
        raise ValueError("label count does not match the manifest")
    jsonl_records = [
        _jsonl_record(json.loads(line))
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(jsonl_records) != manifest["unit_count"]:
        raise ValueError("JSONL label count does not match the manifest")
    seen: set[str] = set()
    for csv_record, jsonl_record in zip(csv_records, jsonl_records, strict=True):
        _validate_record(csv_record, manifest, "CSV")
        _validate_record(jsonl_record, manifest, "JSONL")
        if csv_record["unit_id"] in seen:
            raise ValueError("duplicate unit ID")
        seen.add(csv_record["unit_id"])
        _compare_records(csv_record, jsonl_record)
    source_manifest_path, source_manifest = _load_source_manifest(
        manifest, require_full_snapshot=enforce_t2_2_contract
    )
    _validate_source_artifacts(
        manifest,
        source_manifest_path,
        source_manifest,
        require_full_snapshot=enforce_t2_2_contract,
    )
    _validate_source_snapshot(csv_records, manifest)
    source_index = _load_source_index(manifest, require_full_snapshot=enforce_t2_2_contract)
    _validate_split_independence(csv_records, manifest, source_index)
    _validate_selection(
        csv_records,
        manifest,
        source_index,
        enforce_t2_2_contract=enforce_t2_2_contract,
    )
    return {
        "validated_units": len(csv_records),
        "source_snapshot_id": manifest["source_snapshot_id"],
        "split_role": manifest["split_role"],
        "cutoff": manifest["source_cutoff_timestamp"],
        "model_id": manifest["model_id"],
        "prompt_version": manifest["prompt_version"],
        "model_hash": manifest["model_hash"],
    }


def validate_pseudo_test_artifacts(
    labels_path: Path, jsonl_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Validate the strict, frozen T2.2 pseudo-test contract."""

    return _validate_pseudo_test_artifacts(
        labels_path,
        jsonl_path,
        manifest_path,
        enforce_t2_2_contract=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            validate_pseudo_test_artifacts(arguments.labels, arguments.jsonl, arguments.manifest),
            indent=2,
        )
    )
