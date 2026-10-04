import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from scripts.build_llm_pseudo_test import select_rows
from scripts.validate_llm_pseudo_test import (
    _validate_pseudo_test_artifacts,
    validate_pseudo_test_artifacts,
)


def _validate_generic_fixture(fixture: dict[str, Path]) -> dict[str, object]:
    return _validate_pseudo_test_artifacts(
        fixture["labels"],
        fixture["jsonl"],
        fixture["manifest"],
        enforce_t2_2_contract=False,
    )


def test_select_rows_excludes_pilot_reviews_and_duplicate_groups() -> None:
    reviews = [
        {
            "review_id": "pilot",
            "item_id": "i1",
            "text": "pilot text that is long enough",
            "timestamp": 1,
            "duplicate_group": "g1",
        },
        {
            "review_id": "duplicate",
            "item_id": "i1",
            "text": "duplicate text that is long enough",
            "timestamp": 2,
            "duplicate_group": "g1",
        },
        {
            "review_id": "held-out",
            "item_id": "i2",
            "text": "held out text that is long enough for selection",
            "timestamp": 3,
            "duplicate_group": "g2",
        },
    ]
    interactions = [
        {"review_id": "pilot", "user_id": "u1", "rating": 5.0, "timestamp": 1},
        {"review_id": "duplicate", "user_id": "u2", "rating": 5.0, "timestamp": 2},
        {"review_id": "held-out", "user_id": "u3", "rating": 5.0, "timestamp": 3},
    ]

    selected = select_rows(
        reviews,
        interactions,
        pilot_review_ids={"pilot"},
        pilot_group_keys={("i1", "g1")},
        cutoff_timestamp_ms=10,
        limit=1,
        rating_quota=1,
        minimum_text_length=10,
    )

    assert [row["review_id"] for row in selected] == ["held-out"]


def test_select_rows_rejects_post_cutoff_reviews() -> None:
    with pytest.raises(ValueError, match="eligible"):
        select_rows(
            [
                {
                    "review_id": "future",
                    "item_id": "i1",
                    "text": "a sufficiently long review text",
                    "timestamp": 10,
                    "duplicate_group": "g1",
                }
            ],
            [{"review_id": "future", "user_id": "u1", "rating": 5.0, "timestamp": 10}],
            pilot_review_ids=set(),
            pilot_group_keys=set(),
            cutoff_timestamp_ms=10,
            limit=1,
            rating_quota=1,
            minimum_text_length=10,
        )


def test_select_rows_strict_mode_rejects_missing_rating_bucket() -> None:
    with pytest.raises(ValueError, match="rating bucket"):
        select_rows(
            [
                {
                    "review_id": "only-one",
                    "item_id": "i1",
                    "text": "a sufficiently long review text",
                    "timestamp": 1,
                    "duplicate_group": "g1",
                }
            ],
            [{"review_id": "only-one", "user_id": "u1", "rating": 1.0, "timestamp": 1}],
            pilot_review_ids=set(),
            pilot_group_keys=set(),
            cutoff_timestamp_ms=10,
            require_exact_rating_quota=True,
        )


def test_validator_accepts_frozen_pseudo_test_fixture(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    summary = _validate_generic_fixture(fixture)

    assert summary["validated_units"] == 1
    assert summary["split_role"] == "llm_pseudo_test"


def test_validator_rejects_pilot_overlap(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    pilot = [{"review_id": "held-out", "item_id": "i1", "text": "same"}]
    fixture["pilot"].write_text("\n".join(json.dumps(row) for row in pilot) + "\n")
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["pilot_artifact_sha256"] = hashlib.sha256(fixture["pilot"].read_bytes()).hexdigest()
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="pilot review overlap"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_unfrozen_manifest(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["frozen"] = False
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="frozen"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_tuning_use_in_frozen_manifest(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["usage_restrictions"]["allowed_uses"] = ["prompt_tuning"]
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="allow tuning"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_wrong_t2_2_cardinality(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["selection"]["contract"] = "t2.2-full-snapshot-v1"
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="100"):
        validate_pseudo_test_artifacts(fixture["labels"], fixture["jsonl"], fixture["manifest"])


def test_validator_rejects_source_artifact_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    source_manifest = json.loads(fixture["source_manifest"].read_text())
    source_path = Path(source_manifest["artifact_paths"]["review_texts"])
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "held-out",
                    "item_id": "i1",
                    "text": "tampered source",
                    "timestamp": 1,
                    "duplicate_group": "g1",
                }
            ]
        ),
        source_path,
    )

    with pytest.raises(ValueError, match="source artifact hash"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_source_cutoff_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    source_manifest = json.loads(fixture["source_manifest"].read_text())
    source_manifest["cutoffs"] = {"t0": "1970-01-01T00:00:11Z"}
    fixture["source_manifest"].write_text(json.dumps(source_manifest))

    with pytest.raises(ValueError, match="cutoff"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_pilot_artifact_hash_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    fixture["pilot"].write_text(json.dumps({"review_id": "different"}) + "\n")

    with pytest.raises(ValueError, match="pilot artifact hash"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_boolean_temperature(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["temperature"] = True
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="temperature"):
        _validate_generic_fixture(fixture)


def test_validator_rejects_negative_temperature(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["temperature"] = -1.0
    fixture["manifest"].write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="temperature"):
        _validate_generic_fixture(fixture)


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    source_text = "The graphics are bright and the gameplay is fun."
    source_hash = __import__("hashlib").sha256(source_text.encode()).hexdigest()
    source_reviews = tmp_path / "review_texts.parquet"
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "review_id": "held-out",
                    "item_id": "i1",
                    "text": source_text,
                    "timestamp": 1,
                    "duplicate_group": "g1",
                }
            ]
        ),
        source_reviews,
    )
    source_interactions = tmp_path / "interactions.parquet"
    parquet.write_table(
        pa.Table.from_pylist(
            [{"review_id": "held-out", "user_id": "u1", "rating": 5.0, "timestamp": 1}]
        ),
        source_interactions,
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snapshot-1",
                "dataset_hash": "a" * 64,
                "cutoffs": {"t0": "1970-01-01T00:00:10Z"},
                "artifact_paths": {
                    "review_texts": str(source_reviews),
                    "interactions": str(source_interactions),
                },
                "artifact_sha256": {
                    "review_texts": hashlib.sha256(source_reviews.read_bytes()).hexdigest(),
                    "interactions": hashlib.sha256(source_interactions.read_bytes()).hexdigest(),
                },
            }
        )
    )
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(json.dumps({"review_id": "pilot", "item_id": "i9"}) + "\n")
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "selection_rank": 1,
                "review_id": "held-out",
                "item_id": "i1",
                "duplicate_group": "g1",
                "rating": 5.0,
                "timestamp": 1,
                "source_text_sha256": source_hash,
                "selection_seed": 2202,
            }
        )
        + "\n"
    )
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Annotate explicit game claims.")
    config = tmp_path / "config.toml"
    config.write_text("[test]\nvalue = 1\n")
    restrictions = {
        "allowed_uses": ["aspect_consistency", "sentiment_consistency", "evidence_consistency"],
        "prohibited_uses": [
            "prompt_selection",
            "threshold_tuning",
            "model_tuning",
            "hyperparameter_tuning",
            "recommendation_metrics",
            "human_gold_claim",
            "ground_truth_claim",
        ],
    }
    label = {
        "aspect": "graphics",
        "polarity": "positive",
        "start": source_text.index("graphics"),
        "end": source_text.index("graphics") + len("graphics"),
        "evidence": "graphics",
    }
    record = {
        "schema_version": 2,
        "unit_id": "t2.2-pseudo-001",
        "review_id": "held-out",
        "item_id": "i1",
        "source_snapshot_id": "snapshot-1",
        "source_cutoff_timestamp": "1970-01-01T00:00:10Z",
        "split_role": "llm_pseudo_test",
        "label_status": "llm_pseudo_test",
        "model_id": "gpt-test",
        "model_revision": "unavailable",
        "prompt_version": "t2.2-test-v1",
        "temperature": "unavailable",
        "seed": "unavailable",
        "generated_at": "2026-10-04T00:00:00Z",
        "source_text_sha256": source_hash,
        "timestamp": 1,
        "unit_start": 0,
        "unit_end": len(source_text),
        "unit_text": source_text,
        "text": source_text,
        "source_snapshot_dataset_hash": "a" * 64,
        "configuration_hash": "b" * 64,
        "model_hash": "c" * 64,
        "labels": [label],
        "evidence_offsets": [{"start": label["start"], "end": label["end"]}],
        "out_of_scope": False,
        "needs_review": False,
        "usage_restrictions": restrictions,
        "provenance": {},
    }
    record["provenance"] = {
        **{
            field: record[field]
            for field in (
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
                "source_snapshot_dataset_hash",
                "configuration_hash",
                "model_hash",
            )
        },
        "usage_restrictions": restrictions,
    }
    model_payload = {
        key: record[key]
        for key in ("model_id", "model_revision", "prompt_version", "temperature", "seed")
    }
    model_hash = hashlib.sha256(
        json.dumps(model_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config_hash = hashlib.sha256(str(config).encode() + b"\0[test]\nvalue = 1\n").hexdigest()
    record["configuration_hash"] = config_hash
    record["model_hash"] = model_hash
    record["provenance"]["configuration_hash"] = config_hash
    record["provenance"]["model_hash"] = model_hash
    jsonl = tmp_path / "labels.jsonl"
    jsonl.write_text(json.dumps(record) + "\n")
    csv_path = tmp_path / "labels.csv"
    row = {key: value for key, value in record.items() if key != "provenance"}
    for key in ("labels", "evidence_offsets", "usage_restrictions"):
        row[f"{key}_json"] = json.dumps(row.pop(key), separators=(",", ":"))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest = {
        "schema_version": 2,
        "artifact": str(csv_path),
        "jsonl_artifact": str(jsonl),
        "selection_artifact": str(selection),
        "unit_count": 1,
        "source_dataset": "Amazon Reviews 2023",
        "source_provider": "McAuley Lab",
        "source_snapshot_id": "snapshot-1",
        "source_snapshot_dataset_hash": "a" * 64,
        "source_cutoff_timestamp": "1970-01-01T00:00:10Z",
        "split_role": "llm_pseudo_test",
        "label_status": "llm_pseudo_test",
        "frozen": True,
        "frozen_at": "2026-10-04T00:00:00Z",
        "model_id": "gpt-test",
        "model_revision": "unavailable",
        "prompt_version": "t2.2-test-v1",
        "temperature": "unavailable",
        "seed": "unavailable",
        "generated_at": "2026-10-04T00:00:00Z",
        "source_text_hash_field": "source_text_sha256",
        "required_record_fields": [],
        "usage_restrictions": restrictions,
        "recommendation_test": False,
        "annotation_covers_entire_corpus": False,
        "configuration_files": [str(config)],
        "configuration_hash": config_hash,
        "model_hash": model_hash,
        "prompt_file": str(prompt),
        "prompt_hash": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "pilot_artifact": str(pilot),
        "source_snapshot_manifest": str(source_manifest),
        "hash_definition": {
            "source_snapshot_dataset_hash": "dataset hash from source manifest",
            "configuration_hash": "sha256 of listed configuration paths and bytes",
            "model_hash": "sha256 of canonical model provenance",
        },
        "selection": {
            "seed": 2202,
            "minimum_text_length": 10,
            "rating_quota": 1,
            "group_keys": ["item_id", "duplicate_group"],
        },
        "artifact_sha256": {
            "labels": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "jsonl": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
            "selection": hashlib.sha256(selection.read_bytes()).hexdigest(),
        },
        "pilot_artifact_sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "labels": csv_path,
        "jsonl": jsonl,
        "manifest": manifest_path,
        "pilot": pilot,
        "source_manifest": source_manifest,
    }
