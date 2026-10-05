from datetime import UTC, datetime, timedelta

import pytest

from trustrec.explanations.evidence import (
    AggregatedEvidence,
    aggregate_evidence,
    build_user_aspect_weights,
    compute_aspect_priors,
    personalized_aspect_score,
    review_weight,
)


def test_aggregate_evidence_applies_confidence_duplicate_recency_and_shrinkage() -> None:
    cutoff = datetime(2020, 1, 11, tzinfo=UTC)
    rows = [
        {
            "review_id": "r1",
            "item_id": "i1",
            "aspect": "story",
            "sentiment": "positive",
            "confidence": 1.0,
            "timestamp": datetime(2020, 1, 10, tzinfo=UTC),
            "author_id": "a1",
            "duplicate_group": "d1",
        },
        {
            "review_id": "r2",
            "item_id": "i1",
            "aspect": "story",
            "sentiment": "negative",
            "confidence": 0.5,
            "timestamp": datetime(2020, 1, 9, tzinfo=UTC),
            "author_id": "a2",
            "duplicate_group": "d1",
        },
    ]

    assert review_weight(
        rows[0], cutoff, recency_half_life_days=10, duplicate_group_size=2
    ) == pytest.approx(0.5 * 2 ** (-1 / 10))
    profile = aggregate_evidence(
        rows,
        cutoff_timestamp=cutoff,
        prior_scores={"story": 0.2},
        shrinkage_lambda=1,
        recency_half_life_days=10,
    )[("i1", "story")]

    # Both duplicate-group members receive 1/2. The second review is older.
    first_weight = 0.5 * 2 ** (-1 / 10)
    second_weight = 0.25 * 2 ** (-2 / 10)
    total = first_weight + second_weight
    expected = (0.2 + first_weight - second_weight) / (1 + total)
    assert profile.score == pytest.approx(expected)
    assert profile.support == pytest.approx(total / (1 + total))
    assert profile.author_count == 2
    assert profile.effective_sample_size == pytest.approx(
        total**2 / (first_weight**2 + second_weight**2)
    )
    assert profile.positive_mass == pytest.approx(first_weight)
    assert profile.negative_mass == pytest.approx(second_weight)


def test_each_review_contributes_once_after_clause_aggregation() -> None:
    rows = [
        {
            "review_id": "r1",
            "item_id": "i1",
            "aspect": "gameplay",
            "sentiment": "positive",
            "confidence": 1.0,
            "timestamp": 1,
            "author_id": "a1",
        },
        {
            "review_id": "r1",
            "item_id": "i1",
            "aspect": "gameplay",
            "sentiment": "negative",
            "confidence": 1.0,
            "timestamp": 1,
            "author_id": "a1",
        },
    ]

    profile = aggregate_evidence(rows, cutoff_timestamp=2, shrinkage_lambda=1)[("i1", "gameplay")]
    assert profile.total_weight == pytest.approx(1.0)
    assert profile.positive_mass == pytest.approx(0.5)
    assert profile.negative_mass == pytest.approx(0.5)
    assert profile.score == pytest.approx(0.0)
    assert profile.conflicting is True


def test_future_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="before cutoff"):
        aggregate_evidence(
            [
                {
                    "review_id": "r1",
                    "item_id": "i1",
                    "aspect": "value",
                    "sentiment": "positive",
                    "timestamp": 10,
                }
            ],
            cutoff_timestamp=10,
        )


def test_missing_evidence_keeps_prior_and_zero_support() -> None:
    profile = aggregate_evidence(
        [],
        cutoff_timestamp=10,
        prior_scores={"story": 0.35},
        items=["i1"],
        aspects=["story"],
    )[("i1", "story")]
    assert profile.score == pytest.approx(0.35)
    assert profile.support == 0.0
    assert profile.effective_sample_size == 0.0


def test_priors_and_user_weights_are_smoothed_and_normalized() -> None:
    cutoff = datetime(2020, 1, 2, tzinfo=UTC)
    rows = [
        {
            "review_id": "r1",
            "item_id": "i1",
            "aspect": "story",
            "sentiment": "positive",
            "confidence": 1.0,
            "timestamp": cutoff - timedelta(days=1),
            "user_id": "u1",
        },
        {
            "review_id": "r2",
            "item_id": "i2",
            "aspect": "value",
            "sentiment": "negative",
            "confidence": 1.0,
            "timestamp": cutoff - timedelta(days=1),
            "user_id": "u2",
        },
    ]
    priors = compute_aspect_priors(rows, cutoff_timestamp=cutoff, aspects=["story", "value"])
    assert priors == {"story": pytest.approx(1.0), "value": pytest.approx(-1.0)}
    weights = build_user_aspect_weights(
        rows,
        user_id="u1",
        aspects=["story", "value"],
        prior_distribution={"story": 0.25, "value": 0.75},
        alpha=2,
        cutoff_timestamp=cutoff,
    )
    assert weights["story"] == pytest.approx(0.5)
    assert weights["value"] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_personalized_aspect_score_uses_profile_prior_for_deviation() -> None:
    profile = AggregatedEvidence(
        item_id="i1",
        aspect="story",
        score=0.6,
        prior_score=0.4,
        support=0.8,
        total_weight=4.0,
        positive_mass=3.0,
        negative_mass=0.0,
        neutral_mass=1.0,
        author_count=3,
        effective_sample_size=2.5,
        review_count=3,
    )
    score, support = personalized_aspect_score(
        {"story": 1.0},
        {("i1", "story"): profile},
        item_id="i1",
    )
    assert score == pytest.approx(0.2)
    assert support == pytest.approx(0.8)
