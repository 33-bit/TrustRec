import json
from pathlib import Path


def test_required_directory_contract_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "src/trustrec",
        "tests",
        "configs",
        "configs/experiments",
        "docs/adr",
        "docs/design",
        "docs/specs",
        "docs/plans",
        "docs/tasks",
        "schemas",
        "reports",
    ):
        assert (root / relative).is_dir(), relative


def test_annotation_schema_requires_llm_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas/annotation-record.schema.json").read_text())
    required = set(schema["required"])
    assert {
        "schema_version",
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
        "evidence_offsets",
        "usage_restrictions",
    } <= required


def test_t2_2_freeze_artifacts_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "docs/tasks/t2_2-llm-pseudo-test.md",
        "docs/tasks/t2_2_llm_pseudo_test.manifest.json",
        "docs/tasks/t2_2_llm_pseudo_test.jsonl",
        "docs/tasks/t2_2_llm_pseudo_test.csv",
        "scripts/build_llm_pseudo_test.py",
        "scripts/validate_llm_pseudo_test.py",
    ):
        assert (root / relative).is_file(), relative


def test_annotation_schema_records_unavailable_sampling_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas/annotation-record.schema.json").read_text())

    assert {"const": "unavailable"} in schema["properties"]["temperature"]["oneOf"]
    assert {"const": "unavailable"} in schema["properties"]["seed"]["oneOf"]
