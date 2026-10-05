from datetime import UTC, datetime

from trustrec.explanations.claims import ClaimThresholds, select_explanation
from trustrec.explanations.evidence import aggregate_evidence
from trustrec.explanations.faithfulness import audit_faithfulness


def test_audit_removes_cited_reviews_and_keeps_other_score_parts_fixed() -> None:
    cutoff = datetime(2020, 1, 10, tzinfo=UTC)
    rows = [
        {
            "review_id": "r1",
            "item_id": "i1",
            "aspect": "story",
            "sentiment": "positive",
            "confidence": 1.0,
            "timestamp": datetime(2020, 1, 9, tzinfo=UTC),
            "author_id": "a1",
            "text": "The story is clear.",
            "evidence": "The story is clear.",
            "start": 0,
            "end": 19,
        },
        {
            "review_id": "r2",
            "item_id": "i1",
            "aspect": "gameplay",
            "sentiment": "negative",
            "confidence": 0.2,
            "timestamp": datetime(2020, 1, 8, tzinfo=UTC),
            "author_id": "a2",
            "text": "The story is thin.",
            "evidence": "The story is thin.",
            "start": 0,
            "end": 18,
        },
    ]
    profiles = aggregate_evidence(rows, cutoff_timestamp=cutoff, shrinkage_lambda=1)
    explanation = select_explanation(
        item_id="i1",
        user_id="u1",
        snapshot_id="snap-1",
        profiles=profiles,
        evidence_rows=rows,
        user_aspect_weights={"story": 1.0},
        cutoff_timestamp=cutoff,
        thresholds=ClaimThresholds(min_distinct_authors=1, min_effective_sample_size=1.0),
    )

    result = audit_faithfulness(
        recommendation_id="rec-1",
        user_id="u1",
        item_id="i1",
        snapshot_id="snap-1",
        explanation=explanation,
        evidence_rows=rows,
        cutoff_timestamp=cutoff,
        score_parts={"mf": 0.2, "graph": 0.1, "aspect": 0.6},
        user_aspect_weights={"story": 1.0},
        prior_scores={"story": 0.0},
        shrinkage_lambda=1,
        seed=7,
    )

    assert result.support_status == "supported"
    assert result.removed_review_ids == ("r1",)
    assert result.original_contributions["mf"] == 0.2
    assert result.recomputed_contributions["mf"] == 0.2
    assert result.original_contributions["aspect"] != result.recomputed_contributions["aspect"]
    assert result.original_score != result.recomputed_score
    assert result.random_removed_review_ids
    assert result.audit_result in {"faithful", "not_faithful"}
    assert result.normalization_fixed is True


def test_audit_records_abstention_without_claimed_evidence() -> None:
    result = audit_faithfulness(
        recommendation_id="rec-2",
        user_id="u1",
        item_id="i1",
        snapshot_id="snap-1",
        explanation={
            "status": "insufficient_support",
            "claims": [],
            "refusal_reasons": ["insufficient_support"],
        },
        evidence_rows=[],
        cutoff_timestamp=10,
        score_parts={"mf": 0.4, "aspect": 0.2},
    )

    assert result.audit_result == "abstained"
    assert result.support_status == "insufficient_support"
    assert result.recomputed_score == result.original_score
    assert result.removed_review_ids == ()
