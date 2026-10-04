"""Validate the LLM development-pilot records and their manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "docs/tasks/t2_1_llm_review.csv"
JSONL = ROOT / "docs/tasks/t2_1_llm_review.jsonl"
MANIFEST = ROOT / "docs/tasks/t2_1_llm_review.manifest.json"
ASPECTS = {
    "gameplay",
    "story",
    "graphics",
    "performance",
    "controls",
    "multiplayer",
    "content_replay",
    "value",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


def parse_cutoff(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("the cutoff must include a time zone")
    return round(timestamp.timestamp() * 1000)


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} is not an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a time zone")
    return timestamp


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field} must be a boolean")


def _parse_json(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON") from error


def _model_hash(manifest: dict[str, Any]) -> str:
    payload = {
        field: manifest[field]
        for field in ("model_id", "model_revision", "prompt_version", "temperature", "seed")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _configuration_hash(manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for relative in manifest["configuration_files"]:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"configuration file does not exist: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "artifact",
        "jsonl_artifact",
        "unit_count",
        "source_snapshot_id",
        "source_snapshot_dataset_hash",
        "source_cutoff_timestamp",
        "split_role",
        "label_status",
        "frozen",
        "model_id",
        "model_revision",
        "prompt_version",
        "temperature",
        "seed",
        "generated_at",
        "configuration_files",
        "configuration_hash",
        "model_hash",
        "hash_definition",
        "required_record_fields",
        "usage_restrictions",
        "recommendation_test",
        "annotation_covers_entire_corpus",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("manifest lacks required fields: " + ", ".join(missing))
    if manifest["schema_version"] != 2:
        raise ValueError("manifest must use schema version 2")
    if manifest["split_role"] != "development_pilot":
        raise ValueError("manifest must describe a development_pilot")
    if manifest["label_status"] != "llm_silver":
        raise ValueError("manifest must describe llm_silver labels")
    if manifest["frozen"] is not False:
        raise ValueError("development pilot manifest cannot be frozen")
    if not isinstance(manifest["unit_count"], int) or manifest["unit_count"] <= 0:
        raise ValueError("manifest unit_count must be a positive integer")
    _parse_timestamp(manifest["source_cutoff_timestamp"], "source_cutoff_timestamp")
    _parse_timestamp(manifest["generated_at"], "generated_at")
    if (
        not isinstance(manifest["configuration_files"], list)
        or not manifest["configuration_files"]
        or not all(isinstance(path, str) and path for path in manifest["configuration_files"])
    ):
        raise ValueError("manifest configuration_files must be a non-empty list")
    for field in ("source_snapshot_dataset_hash", "configuration_hash", "model_hash"):
        if not isinstance(manifest[field], str) or not SHA256_RE.fullmatch(manifest[field]):
            raise ValueError(f"manifest {field} must be a lowercase SHA-256 hash")
    if manifest["model_hash"] != _model_hash(manifest):
        raise ValueError("manifest model_hash does not match the model provenance")
    if manifest["configuration_hash"] != _configuration_hash(manifest):
        raise ValueError("manifest configuration_hash does not match the configuration files")
    if not isinstance(manifest["hash_definition"], dict) or set(manifest["hash_definition"]) != {
        "source_snapshot_dataset_hash",
        "configuration_hash",
        "model_hash",
    }:
        raise ValueError("manifest hash_definition is incomplete")
    if manifest["required_record_fields"] != [
        "schema_version",
        "unit_id",
        "review_id",
        "item_id",
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
        "source_text_sha256",
        "timestamp",
        "unit_start",
        "unit_end",
        "unit_text",
        "source_snapshot_dataset_hash",
        "configuration_hash",
        "model_hash",
        "labels",
        "evidence_offsets",
        "usage_restrictions",
    ]:
        raise ValueError("manifest required_record_fields are incomplete")
    restrictions = manifest["usage_restrictions"]
    if not isinstance(restrictions, dict) or set(restrictions) != {
        "allowed_uses",
        "prohibited_uses",
    }:
        raise ValueError("manifest usage_restrictions are incomplete")
    if not all(isinstance(value, list) for value in restrictions.values()):
        raise ValueError("manifest usage restrictions must be lists")


def _csv_record(row: dict[str, str]) -> dict[str, Any]:
    record = dict(row)
    for field in ("schema_version", "seed", "timestamp", "unit_start", "unit_end"):
        try:
            record[field] = int(record[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"CSV field {field} is not an integer") from error
    try:
        record["temperature"] = float(record["temperature"])
    except (TypeError, ValueError) as error:
        raise ValueError("CSV field temperature is not a number") from error
    for field in ("labels", "evidence_offsets", "usage_restrictions"):
        csv_field = f"{field}_json"
        record[field] = _parse_json(record.pop(csv_field), csv_field)
    record["out_of_scope"] = _parse_bool(record["out_of_scope"], "CSV out_of_scope")
    record["needs_review"] = _parse_bool(record["needs_review"], "CSV needs_review")
    return record


def _expected_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    return {field: manifest[field] for field in PROVENANCE_FIELDS}


def _validate_record(record: dict[str, Any], manifest: dict[str, Any], source: str) -> None:
    required = {
        "schema_version",
        "unit_id",
        "review_id",
        "item_id",
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
        "source_text_sha256",
        "timestamp",
        "unit_start",
        "unit_end",
        "unit_text",
        "text",
        "source_snapshot_dataset_hash",
        "configuration_hash",
        "model_hash",
        "labels",
        "evidence_offsets",
        "out_of_scope",
        "needs_review",
        "usage_restrictions",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{source} record lacks required fields: {', '.join(missing)}")
    if record["schema_version"] != 2:
        raise ValueError(f"{source} record must use schema version 2")
    expected = _expected_provenance(manifest)
    for field, expected_value in expected.items():
        if record[field] != expected_value:
            raise ValueError(f"{source} provenance mismatch for {field}")
    if not isinstance(record["text"], str):
        raise ValueError(f"{source} text must be a string")
    if not isinstance(record["unit_text"], str):
        raise ValueError(f"{source} unit_text must be a string")
    try:
        unit_start, unit_end = int(record["unit_start"]), int(record["unit_end"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} unit offsets are invalid") from error
    if not 0 <= unit_start < unit_end <= len(record["text"]):
        raise ValueError(f"{source} unit offsets exceed the source text")
    if record["text"][unit_start:unit_end] != record["unit_text"]:
        raise ValueError(f"{source} unit_text does not match source offsets")
    _parse_timestamp(record["source_cutoff_timestamp"], f"{source} source_cutoff_timestamp")
    _parse_timestamp(record["generated_at"], f"{source} generated_at")
    if int(record["timestamp"]) >= parse_cutoff(manifest["source_cutoff_timestamp"]):
        raise ValueError(f"{source} contains a post-cutoff review")
    if not isinstance(record["source_text_sha256"], str) or not SHA256_RE.fullmatch(
        record["source_text_sha256"]
    ):
        raise ValueError(f"{source} source_text_sha256 is invalid")
    if record["source_text_sha256"] != hashlib.sha256(record["text"].encode()).hexdigest():
        raise ValueError(f"{source} source text hash mismatch for {record['unit_id']}")
    if not isinstance(record["out_of_scope"], bool) or not isinstance(record["needs_review"], bool):
        raise ValueError(f"{source} scope flags must be booleans")
    if not isinstance(record["labels"], list) or not isinstance(record["evidence_offsets"], list):
        raise ValueError(f"{source} labels and evidence_offsets must be lists")
    if len(record["evidence_offsets"]) != len(record["labels"]):
        raise ValueError(f"{source} evidence offset count does not match labels")
    if record["out_of_scope"] and record["labels"]:
        raise ValueError(f"{source} out-of-scope rows cannot contain aspect labels")
    if record["usage_restrictions"] != manifest["usage_restrictions"]:
        raise ValueError(f"{source} usage restrictions do not match the manifest")
    for label, offset in zip(record["labels"], record["evidence_offsets"], strict=True):
        if not isinstance(label, dict) or label.get("aspect") not in ASPECTS:
            raise ValueError(f"{source} invalid aspect")
        if label.get("polarity") not in {"positive", "negative", "neutral"}:
            raise ValueError(f"{source} invalid polarity")
        try:
            start, end = int(label["start"]), int(label["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source} label offsets are invalid") from error
        if not isinstance(offset, dict) or offset.get("start") != start or offset.get("end") != end:
            raise ValueError(f"{source} evidence offsets do not match labels")
        if not unit_start <= start < end <= unit_end:
            raise ValueError(f"{source} evidence offsets exceed the source text")
        if not isinstance(label.get("evidence"), str):
            raise ValueError(f"{source} evidence text is missing")
        if record["text"][start:end] != label["evidence"]:
            raise ValueError(f"{source} evidence does not match source offsets")


def _jsonl_record(record: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(record)
    if "provenance" not in parsed or not isinstance(parsed["provenance"], dict):
        raise ValueError("JSONL record lacks provenance")
    provenance = parsed["provenance"]
    for field in PROVENANCE_FIELDS:
        if field not in parsed:
            raise ValueError(f"JSONL provenance lacks {field}")
        if field not in provenance:
            raise ValueError(f"JSONL provenance lacks {field}")
    if "usage_restrictions" not in parsed:
        raise ValueError("JSONL provenance lacks usage_restrictions")
    if "usage_restrictions" not in provenance:
        raise ValueError("JSONL provenance lacks usage_restrictions")
    if "source_text_sha256" not in parsed:
        raise ValueError("JSONL provenance lacks source_text_sha256")
    if "source_text_sha256" not in provenance:
        raise ValueError("JSONL provenance lacks source_text_sha256")
    if "timestamp" not in parsed:
        raise ValueError("JSONL record lacks timestamp")
    expected_provenance = {
        **{field: parsed[field] for field in PROVENANCE_FIELDS},
        "source_text_sha256": parsed["source_text_sha256"],
        "usage_restrictions": parsed["usage_restrictions"],
    }
    if parsed["provenance"] != expected_provenance:
        raise ValueError("JSONL provenance does not match top-level fields")
    return parsed


def _compare_records(csv_record: dict[str, Any], jsonl_record: dict[str, Any]) -> None:
    fields = (
        "schema_version",
        "unit_id",
        "review_id",
        "item_id",
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
        "source_text_sha256",
        "timestamp",
        "unit_start",
        "unit_end",
        "unit_text",
        "text",
        "source_snapshot_dataset_hash",
        "configuration_hash",
        "model_hash",
        "labels",
        "evidence_offsets",
        "out_of_scope",
        "needs_review",
        "usage_restrictions",
    )
    for field in fields:
        if csv_record[field] != jsonl_record[field]:
            raise ValueError(f"CSV and JSONL mismatch for {field} in {csv_record['unit_id']}")


def _validate_source_snapshot(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """Check label text and IDs against an optional source snapshot manifest."""

    source_manifest_value = manifest.get("source_snapshot_manifest")
    if not source_manifest_value:
        return
    source_manifest_path = Path(source_manifest_value)
    if not source_manifest_path.is_absolute():
        source_manifest_path = ROOT / source_manifest_path
    if not source_manifest_path.is_file():
        raise ValueError(f"source snapshot manifest does not exist: {source_manifest_value}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("snapshot_id") != manifest["source_snapshot_id"]:
        raise ValueError("source snapshot ID does not match the label manifest")
    if source_manifest.get("dataset_hash") != manifest["source_snapshot_dataset_hash"]:
        raise ValueError("source snapshot dataset hash does not match the label manifest")
    artifact_value = source_manifest["artifact_paths"]["review_texts"]
    artifact_path = Path(artifact_value)
    if not artifact_path.is_absolute():
        artifact_path = source_manifest_path.parent.parent.parent / artifact_path
    if not artifact_path.is_file():
        raise ValueError(f"source review text artifact does not exist: {artifact_value}")
    needed = {record["review_id"] for record in records}
    source_rows = {}
    for batch in parquet.ParquetFile(artifact_path).iter_batches(
        columns=["review_id", "text"], use_threads=False
    ):
        for row in batch.to_pylist():
            if row["review_id"] in needed:
                source_rows[row["review_id"]] = row["text"]
        if needed <= source_rows.keys():
            break
    for record in records:
        review_id = record["review_id"]
        if review_id not in source_rows:
            raise ValueError(f"source snapshot is missing review {review_id}")
        if source_rows[review_id] != record["text"]:
            raise ValueError(f"source snapshot text mismatch for {review_id}")


def validate_artifacts(
    labels_path: Path = LABELS,
    jsonl_path: Path = JSONL,
    manifest_path: Path = MANIFEST,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    with labels_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != manifest["unit_count"]:
        raise ValueError("label count does not match the manifest")
    csv_records = [_csv_record(row) for row in csv_rows]
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
    _validate_source_snapshot(csv_records, manifest)
    return {
        "validated_units": len(csv_records),
        "jsonl_units": len(jsonl_records),
        "source_snapshot_id": manifest["source_snapshot_id"],
        "cutoff": manifest["source_cutoff_timestamp"],
        "configuration_hash": manifest["configuration_hash"],
        "model_hash": manifest["model_hash"],
    }


def validate(path: Path = LABELS) -> dict[str, Any]:
    jsonl_path = JSONL if path == LABELS else path.with_suffix(".jsonl")
    return validate_artifacts(path, jsonl_path, MANIFEST)


if __name__ == "__main__":
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LABELS
    print(json.dumps(validate(labels_path), indent=2))
