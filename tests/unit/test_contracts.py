from datetime import UTC, datetime

import pytest

from trustrec.schemas import EvidenceRef, RecommendationRecord, SnapshotManifest


def test_snapshot_manifest_serializes_timezone_aware_provenance() -> None:
    manifest = SnapshotManifest(
        snapshot_id="vg-train-001",
        cutoff_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_hash="sha256:abc",
        source_uri="s3://example/reviews.parquet",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        row_counts={"interactions": 10},
    )

    assert manifest.to_dict()["cutoff_timestamp"].endswith("+00:00")


def test_evidence_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        EvidenceRef(
            review_id="r1",
            item_id="i1",
            aspect="gameplay",
            text="Good",
            sentiment="positive",
            confidence=1.1,
            source_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_recommendation_requires_positive_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        RecommendationRecord(
            user_id="u1",
            snapshot_id="s1",
            item_id="i1",
            rank=0,
            total_score=0.5,
            component_scores={"mf": 0.5},
            evidence=(),
            explanation_status="unavailable",
        )
