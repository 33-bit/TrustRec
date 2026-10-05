from datetime import UTC, datetime

import pytest

from trustrec.explanations.claims import ClaimThresholds, select_explanation
from trustrec.explanations.evidence import aggregate_evidence


def _rows() -> list[dict[str, object]]:
    return [
        {
            "review_id": "r1",
            "item_id": "i1",
            "aspect": "story",
            "sentiment": "positive",
            "confidence": 0.95,
            "timestamp": datetime(2020, 1, 9, tzinfo=UTC),
            "author_id": "a1",
            "text": "The story is excellent.",
            "evidence": "The story is excellent.",
            "start": 0,
            "end": 23,
        },
        {
            "review_id": "r2",
            "item_id": "i1",
            "aspect": "story",
            "sentiment": "negative",
            "confidence": 0.90,
            "timestamp": datetime(2020, 1, 8, tzinfo=UTC),
            "author_id": "a2",
            "text": "The story is confusing.",
            "evidence": "The story is confusing.",
            "start": 0,
            "end": 23,
        },
        {
            "review_id": "r3",
            "item_id": "i1",
            "aspect": "gameplay",
            "sentiment": "positive",
            "confidence": 0.95,
            "timestamp": datetime(2020, 1, 9, tzinfo=UTC),
            "author_id": "a3",
            "text": "Combat feels smooth.",
            "evidence": "Combat feels smooth.",
            "start": 0,
            "end": 20,
        },
    ]


def test_select_explanation_cites_offsets_and_records_dominant_score_part() -> None:
    rows = [_rows()[0], _rows()[2]]
    profiles = aggregate_evidence(
        rows,
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        shrinkage_lambda=1,
    )

    explanation = select_explanation(
        item_id="i1",
        user_id="u1",
        snapshot_id="snap-1",
        profiles=profiles,
        evidence_rows=rows,
        user_aspect_weights={"story": 0.8, "gameplay": 0.2},
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        score_parts={"mf": 0.7, "graph": 0.2, "aspect": 0.3},
        thresholds=ClaimThresholds(
            min_support=0.1,
            min_distinct_authors=1,
            min_effective_sample_size=1.0,
            min_confidence=0.5,
        ),
        max_claims=1,
    )

    assert explanation.status == "supported"
    assert explanation.dominant_score_part == "mf"
    assert explanation.dominant_score_text == "The mf score part is the largest contribution."
    assert len(explanation.claims) == 1
    claim = explanation.claims[0]
    assert claim.status == "supported"
    assert claim.aspect == "story"
    assert claim.sentiment_direction == "positive"
    assert claim.evidence[0].review_id == "r1"
    assert claim.evidence[0].start_offset == 0
    assert claim.evidence[0].end_offset == 23
    assert claim.evidence[0].text == "The story is excellent."


def test_select_explanation_abstains_when_support_is_weak() -> None:
    rows = [_rows()[0]]
    profiles = aggregate_evidence(
        rows,
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        shrinkage_lambda=1,
    )

    explanation = select_explanation(
        item_id="i1",
        profiles=profiles,
        evidence_rows=rows,
        user_aspect_weights={"story": 1.0},
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        thresholds=ClaimThresholds(min_distinct_authors=2),
    )

    assert explanation.status == "insufficient_support"
    assert explanation.claims[0].status == "insufficient_support"
    assert explanation.claims[0].claim is None
    assert "insufficient" in explanation.claims[0].refusal_message.lower()
    assert "distinct_authors" in explanation.refusal_reasons[0]


def test_select_explanation_marks_conflicting_evidence() -> None:
    rows = _rows()[:2]
    profiles = aggregate_evidence(
        rows,
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        shrinkage_lambda=1,
    )

    explanation = select_explanation(
        item_id="i1",
        profiles=profiles,
        evidence_rows=rows,
        user_aspect_weights={"story": 1.0},
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        thresholds=ClaimThresholds(min_distinct_authors=2, min_effective_sample_size=1.0),
    )

    assert explanation.status == "supported"
    assert explanation.claims[0].sentiment_direction == "conflicting"
    assert len(explanation.claims[0].evidence) == 2
    assert "mixed" in explanation.claims[0].claim.lower()


def test_select_explanation_rejects_future_evidence() -> None:
    rows = [_rows()[0]]
    profiles = aggregate_evidence(
        rows,
        cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        shrinkage_lambda=1,
    )

    with pytest.raises(ValueError, match="before cutoff"):
        select_explanation(
            item_id="i1",
            profiles=profiles,
            evidence_rows=[{**rows[0], "timestamp": datetime(2020, 1, 10, tzinfo=UTC)}],
            user_aspect_weights={"story": 1.0},
            cutoff_timestamp=datetime(2020, 1, 10, tzinfo=UTC),
        )
