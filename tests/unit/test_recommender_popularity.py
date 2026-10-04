import pytest

from trustrec.recommenders.contracts import CandidateSet
from trustrec.recommenders.popularity import PopularityRanker, rank_popularity


def test_popularity_uses_positive_pre_cutoff_counts_and_stable_ties() -> None:
    result = rank_popularity(
        [
            {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u2", "item_id": "a", "rating": 4, "timestamp": 2},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 3},
            {"user_id": "u4", "item_id": "b", "rating": 3, "timestamp": 4},
            {"user_id": "u5", "item_id": "c", "rating": 3, "timestamp": 5},
            {"user_id": "u6", "item_id": "future", "rating": 5, "timestamp": 11},
        ],
        CandidateSet("u9", "snap-1", 10, ("a", "b", "c")),
    )

    assert [item.item_id for item in result.items] == ["a", "b", "c"]
    assert [item.total_score for item in result.items] == [2.0, 1.0, 0.0]


def test_popularity_returns_empty_result_for_empty_candidates() -> None:
    result = rank_popularity([], CandidateSet("u1", "snap-1", 10, ()))

    assert result.items == ()
    assert result.fallback_reason == "no_candidates"


def test_popularity_ranker_requires_the_fitted_cutoff() -> None:
    ranker = PopularityRanker().fit(
        [{"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 1}],
        cutoff_timestamp=10,
    )

    result = ranker.rank(CandidateSet("u9", "snap-1", 10, ("a",)))
    assert result.candidate_item_ids == ("a",)
    assert result.model_hash == ranker.model_hash


def test_popularity_rejects_a_different_cutoff() -> None:
    ranker = PopularityRanker().fit(
        [{"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 1}],
        cutoff_timestamp=10,
    )

    with pytest.raises(ValueError, match="cutoff"):
        ranker.rank(CandidateSet("u9", "snap-1", 11, ("a",)))
