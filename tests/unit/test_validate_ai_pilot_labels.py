import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from scripts.validate_ai_pilot_labels import validate_artifacts

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "docs/tasks/t2_1_llm_review.csv"
JSONL = ROOT / "docs/tasks/t2_1_llm_review.jsonl"
MANIFEST = ROOT / "docs/tasks/t2_1_llm_review.manifest.json"


def test_validate_artifacts_checks_csv_jsonl_and_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "review_texts.parquet"
    records = [json.loads(line) for line in JSONL.read_text().splitlines()]
    parquet.write_table(
        pa.Table.from_pylist(
            [{"review_id": record["review_id"], "text": record["text"]} for record in records]
        ),
        source_path,
    )
    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "snapshot_id": records[0]["source_snapshot_id"],
                "dataset_hash": records[0]["source_snapshot_dataset_hash"],
                "artifact_paths": {"review_texts": str(source_path)},
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(MANIFEST.read_text())
    manifest["source_snapshot_manifest"] = str(source_manifest_path)
    manifest_path.write_text(json.dumps(manifest))

    summary = validate_artifacts(LABELS, JSONL, manifest_path)

    assert summary["validated_units"] == 100
    assert summary["jsonl_units"] == 100
    assert summary["source_snapshot_id"] == "video_games-full-d6c4efeb74aa"
    assert len(summary["configuration_hash"]) == 64
    assert len(summary["model_hash"]) == 64


def test_validate_artifacts_rejects_jsonl_provenance_mismatch(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    jsonl = tmp_path / "labels.jsonl"
    manifest = tmp_path / "manifest.json"
    labels.write_bytes(LABELS.read_bytes())
    manifest.write_bytes(MANIFEST.read_bytes())
    records = [json.loads(line) for line in JSONL.read_text().splitlines()]
    records[0]["source_cutoff_timestamp"] = "2021-04-09T19:03:26.065Z"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="JSONL provenance"):
        validate_artifacts(labels, jsonl, manifest)


def test_validate_artifacts_rejects_jsonl_source_hash_mismatch(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    jsonl = tmp_path / "labels.jsonl"
    manifest = tmp_path / "manifest.json"
    labels.write_bytes(LABELS.read_bytes())
    manifest.write_bytes(MANIFEST.read_bytes())
    records = [json.loads(line) for line in JSONL.read_text().splitlines()]
    records[0]["source_text_sha256"] = "0" * 64
    records[0]["provenance"]["source_text_sha256"] = "0" * 64
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="JSONL source text hash mismatch"):
        validate_artifacts(labels, jsonl, manifest)


def test_validate_artifacts_requires_nested_source_text_hash(tmp_path: Path) -> None:
    jsonl = tmp_path / "labels.jsonl"
    records = [json.loads(line) for line in JSONL.read_text().splitlines()]
    records[0]["provenance"].pop("source_text_sha256", None)
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="provenance.*source_text_sha256"):
        validate_artifacts(LABELS, jsonl, MANIFEST)


def test_validate_artifacts_accepts_review_relative_evidence(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    manifest["unit_count"] = 1
    manifest.pop("source_snapshot_manifest", None)
    record = json.loads(JSONL.read_text().splitlines()[0])
    review_text = "Review prefix. The game could not access the game."
    unit_text = "could not access the game"
    unit_start = review_text.index(unit_text)
    record.update(
        text=review_text,
        unit_start=unit_start,
        unit_end=unit_start + len(unit_text),
        unit_text=unit_text,
        source_text_sha256=hashlib.sha256(review_text.encode()).hexdigest(),
        out_of_scope=False,
        needs_review=False,
        labels=[
            {
                "aspect": "performance",
                "polarity": "negative",
                "start": unit_start,
                "end": unit_start + len(unit_text),
                "evidence": "could not access the game",
            }
        ],
        evidence_offsets=[{"start": unit_start, "end": unit_start + len(unit_text)}],
    )
    record["provenance"]["source_text_sha256"] = record["source_text_sha256"]
    row = {k: v for k, v in record.items() if k != "provenance"}
    for key in ("labels", "evidence_offsets", "usage_restrictions"):
        row[f"{key}_json"] = json.dumps(row.pop(key))
    labels, jsonl, manifest_path = (
        tmp_path / "labels.csv",
        tmp_path / "labels.jsonl",
        tmp_path / "manifest.json",
    )
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    jsonl.write_text(json.dumps(record) + "\n")
    manifest_path.write_text(json.dumps(manifest))

    assert validate_artifacts(labels, jsonl, manifest_path)["validated_units"] == 1


def test_validate_artifacts_checks_source_snapshot_text(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    jsonl = tmp_path / "labels.jsonl"
    manifest_path = tmp_path / "manifest.json"
    source_manifest_path = tmp_path / "source-manifest.json"
    source_texts_path = tmp_path / "review_texts.parquet"
    labels.write_bytes(LABELS.read_bytes())
    records = [json.loads(line) for line in JSONL.read_text().splitlines()]
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {"review_id": record["review_id"], "text": "tampered" if i == 0 else record["text"]}
                for i, record in enumerate(records)
            ]
        ),
        source_texts_path,
    )
    source_manifest_path.write_text(
        json.dumps(
            {
                "snapshot_id": records[0]["source_snapshot_id"],
                "dataset_hash": records[0]["source_snapshot_dataset_hash"],
                "artifact_paths": {"review_texts": str(source_texts_path)},
            }
        )
    )
    manifest = json.loads(MANIFEST.read_text())
    manifest["source_snapshot_manifest"] = str(source_manifest_path)
    manifest_path.write_text(json.dumps(manifest))
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="source snapshot text mismatch"):
        validate_artifacts(labels, jsonl, manifest_path)
