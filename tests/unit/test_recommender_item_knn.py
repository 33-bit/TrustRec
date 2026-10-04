import pytest

from trustrec.recommenders.contracts import CandidateSet
from trustrec.recommenders.item_knn import ItemKNNRanker, rank_item_knn


def test_item_knn_scores_candidates_by_cosine_similarity() -> None:
    result = rank_item_knn(
        [
            {"user_id": "u1", "item_id": "h", "rating": 5, "timestamp": 1},
            {"user_id": "u2", "item_id": "h", "rating": 5, "timestamp": 2},
            {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 3},
            {"user_id": "u3", "item_id": "a", "rating": 5, "timestamp": 4},
            {"user_id": "u2", "item_id": "b", "rating": 5, "timestamp": 5},
        ],
        CandidateSet("u1", "snap-1", 10, ("b",), frozenset({"h"})),
    )

    assert result.items[0].item_id == "b"
    assert result.items[0].component_scores["knn"] == pytest.approx(2**-0.5)


def test_item_knn_falls_back_to_popularity_without_positive_history() -> None:
    result = rank_item_knn(
        [
            {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 2},
        ],
        CandidateSet("u1", "snap-1", 10, ("a", "b")),
    )

    assert result.fallback_reason == "no_positive_history"
    assert [item.item_id for item in result.items] == ["a", "b"]


def test_item_knn_neighbor_limit_keeps_strongest_history_edges() -> None:
    result = rank_item_knn(
        [
            {"user_id": "u1", "item_id": "h1", "rating": 5, "timestamp": 1},
            {"user_id": "u1", "item_id": "h2", "rating": 5, "timestamp": 2},
            {"user_id": "u2", "item_id": "h1", "rating": 5, "timestamp": 3},
            {"user_id": "u2", "item_id": "c", "rating": 5, "timestamp": 4},
            {"user_id": "u3", "item_id": "h2", "rating": 5, "timestamp": 5},
        ],
        CandidateSet("u1", "snap-1", 10, ("c",), frozenset({"h1", "h2"})),
        n_neighbors=1,
    )

    assert result.items[0].component_scores["knn"] == pytest.approx(1 / 2**0.5)


def test_item_knn_empty_candidates_keep_the_shared_contract() -> None:
    result = ItemKNNRanker().fit([], cutoff_timestamp=10).rank(CandidateSet("u1", "snap-1", 10, ()))

    assert result.items == ()
    assert result.candidate_item_ids == ()
    assert result.fallback_reason == "no_candidates"
