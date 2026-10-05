import json
from pathlib import Path

from scripts.build_evidence_profiles import build_profiles


def test_build_profiles_records_cutoff_and_hashes(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "review_id": "r1",
                "item_id": "i1",
                "aspect": "story",
                "sentiment": "positive",
                "confidence": 1.0,
                "timestamp": 1,
                "author_id": "a1",
            }
        )
        + "\n"
    )
    output = tmp_path / "profiles.json"
    artifact = build_profiles(
        evidence,
        snapshot_id="snap-1",
        dataset_hash="d" * 64,
        cutoff_timestamp="1970-01-01T00:00:00.010000+00:00",
        output_path=output,
        items=["i1", "i2"],
        aspects=["story"],
    )
    assert artifact["status"] == "complete"
    assert artifact["configuration_hash"]
    assert artifact["source_evidence_sha256"]
    missing = next(row for row in artifact["profiles"] if row["item_id"] == "i2")
    assert missing["score"] == 0.0
    assert missing["support"] == 0.0
