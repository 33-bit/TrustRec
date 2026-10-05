import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.run_explanation_audit import run_audit


def test_run_audit_writes_lineage_and_removal_rows(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(
        json.dumps(
            {
                "review_id": "r1",
                "item_id": "i1",
                "aspect": "story",
                "sentiment": "positive",
                "confidence": 1.0,
                "timestamp": 1578528000000,
                "author_id": "a1",
                "text": "The story is clear.",
                "evidence": "The story is clear.",
                "start": 0,
                "end": 19,
            }
        )
        + "\n"
    )
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "recommendation_id": "rec-1",
                        "user_id": "u1",
                        "item_id": "i1",
                        "snapshot_id": "snap-1",
                        "score_parts": {"mf": 0.2, "aspect": 0.4},
                        "explanation": {
                            "status": "supported",
                            "claims": [
                                {
                                    "item_id": "i1",
                                    "aspect": "story",
                                    "claim": "Item i1 has positive evidence for story.",
                                    "sentiment_direction": "positive",
                                    "status": "supported",
                                    "evidence": [
                                        {
                                            "review_id": "r1",
                                            "item_id": "i1",
                                            "aspect": "story",
                                            "text": "The story is clear.",
                                            "sentiment": "positive",
                                            "confidence": 1.0,
                                            "source_timestamp": 1578528000000,
                                            "start": 0,
                                            "end": 19,
                                        }
                                    ],
                                    "support": 0.5,
                                    "distinct_author_count": 1,
                                    "effective_sample_size": 1.0,
                                    "score_contribution": 0.5,
                                    "conflicting": False,
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )
    output_path = tmp_path / "audit.json"

    artifact = run_audit(
        evidence_path,
        claims_path,
        snapshot_id="snap-1",
        dataset_hash="d" * 64,
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        output_path=output_path,
        shrinkage_lambda=1,
    )

    assert artifact["status"] == "complete"
    assert artifact["snapshot_id"] == "snap-1"
    assert artifact["row_counts"]["audits"] == 1
    assert artifact["audits"][0]["evidence_review_id"] == "r1"
    assert artifact["audits"][0]["normalization_fixed"] is True
    assert json.loads(output_path.read_text()) == artifact
