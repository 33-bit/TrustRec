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
        "status",
        "sources",
        "runs",
        "claims",
        "generated_files",
    } <= set(schema["required"])
    assert {"claim_id", "kind", "statement", "run_id", "evidence"} <= set(
        schema["$defs"]["claim"]["required"]
    )
