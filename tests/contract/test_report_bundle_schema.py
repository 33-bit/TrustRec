from __future__ import annotations

import json
from pathlib import Path


def test_report_bundle_schema_declares_manifest_and_claim_fields() -> None:
    root = Path(__file__).parents[2]
    schema = json.loads((root / "schemas/report-bundle.schema.json").read_text(encoding="utf-8"))

    assert schema["properties"]["status"]["enum"] == ["complete"]
    assert {
        "schema_version",
        "bundle_id",
        "title",
        "status",
        "code_revision",
        "specification_path",
        "specification_sha256",
        "sources",
        "commands",
        "runs",
        "claims",
        "generated_files",
        "checksums_path",
    } <= set(schema["required"])
    assert {
        "status",
        "snapshot_id",
        "dataset_hash",
        "cutoffs",
        "protocol_version",
        "seed",
        "seeds",
        "configuration_hash",
        "model_hashes",
        "code_revision",
    } <= set(schema["$defs"]["run"]["required"])
    assert {"claim_id", "kind", "statement", "run_id", "evidence"} <= set(
        schema["$defs"]["claim"]["required"]
    )
