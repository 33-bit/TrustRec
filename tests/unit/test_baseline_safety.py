import json

import pytest

from trustrec.recommenders import (
    CandidateSet,
    ItemKNNRanker,
    PopularityRanker,
    build_candidate_set,
)


@pytest.mark.parametrize("ranker_type", [PopularityRanker, ItemKNNRanker])
def test_baselines_use_first_event_for_repeated_pairs(ranker_type) -> None:
    rows = [
        {"review_id": "r2", "user_id": "v", "item_id": "a", "rating": 5, "timestamp": 2},
        {"review_id": "r1", "user_id": "v", "item_id": "a", "rating": 2, "timestamp": 1},
        {"review_id": "r3", "user_id": "w", "item_id": "b", "rating": 4, "timestamp": 3},
        {"review_id": "r4", "user_id": "w", "item_id": "b", "rating": 5, "timestamp": 4},
    ]
    candidates = build_candidate_set("new", "snap-1", rows, 10)
    result = ranker_type().fit(rows, 10).rank(candidates)

    assert [(item.item_id, item.total_score) for item in result.items] == [("b", 1.0), ("a", 0.0)]


@pytest.mark.parametrize("ranker_type", [PopularityRanker, ItemKNNRanker])
def test_baselines_reject_reviewed_items_even_when_history_is_omitted(ranker_type) -> None:
    ranker = ranker_type().fit([{"user_id": "u", "item_id": "a", "rating": 1, "timestamp": 1}], 10)

    with pytest.raises(ValueError, match="history"):
        ranker.rank(CandidateSet("u", "snap-1", 10, ("a",)))


@pytest.mark.parametrize("ranker_type", [PopularityRanker, ItemKNNRanker])
def test_baselines_reject_items_outside_pre_cutoff_catalog(ranker_type) -> None:
    ranker = ranker_type().fit([{"user_id": "v", "item_id": "a", "rating": 5, "timestamp": 10}], 10)

    with pytest.raises(ValueError, match="known before the cutoff"):
        ranker.rank(CandidateSet("u", "snap-1", 10, ("a",)))


@pytest.mark.parametrize("ranker_type", [PopularityRanker, ItemKNNRanker])
def test_future_rows_do_not_change_fitted_hash_or_scores(ranker_type) -> None:
    past = [
        {"user_id": "u", "item_id": "h", "rating": 5, "timestamp": 1},
        {"user_id": "v", "item_id": "h", "rating": 5, "timestamp": 2},
        {"user_id": "v", "item_id": "a", "rating": 4, "timestamp": 3},
        {"user_id": "w", "item_id": "b", "rating": 2, "timestamp": 4},
    ]
    future = [
        {"user_id": "u", "item_id": "a", "rating": 5, "timestamp": 10},
        {"user_id": "v", "item_id": "new", "rating": 5, "timestamp": 11},
    ]
    candidates = build_candidate_set("u", "snap-1", past, 10)
    before = ranker_type(seed=7).fit(past, 10).rank(candidates)
    after = ranker_type(seed=7).fit(list(reversed(past + future)), 10).rank(candidates)

    assert before.to_dict() == after.to_dict()
    assert before.candidate_item_ids == ("a", "b")
    assert before.items[0].item_id == "a"


@pytest.mark.parametrize("ranker_type", [PopularityRanker, ItemKNNRanker])
def test_snapshot_bound_models_refuse_a_different_snapshot(ranker_type) -> None:
    rows = [{"user_id": "v", "item_id": "a", "rating": 5, "timestamp": 1}]
    ranker = ranker_type().fit(rows, 10, snapshot_id="snap-1")

    with pytest.raises(ValueError, match="snapshot"):
        ranker.rank(CandidateSet("u", "snap-2", 10, ("a",)))


def test_neighborless_history_reports_popularity_fallback() -> None:
    rows = [
        {"user_id": "u", "item_id": "h", "rating": 5, "timestamp": 1},
        {"user_id": "v", "item_id": "b", "rating": 5, "timestamp": 2},
        {"user_id": "w", "item_id": "b", "rating": 4, "timestamp": 3},
        {"user_id": "x", "item_id": "a", "rating": 5, "timestamp": 4},
    ]
    result = ItemKNNRanker().fit(rows, 10).rank(build_candidate_set("u", "snap-1", rows, 10))

    assert result.fallback_reason == "no_similar_items"
    assert [(item.item_id, item.total_score) for item in result.items] == [("b", 2.0), ("a", 1.0)]
    assert all(item.component_scores["knn"] == 0.0 for item in result.items)


@pytest.mark.parametrize("ranker_type", [PopularityRanker, ItemKNNRanker])
def test_top_k_preserves_complete_candidate_contract(ranker_type) -> None:
    rows = [
        {"user_id": "v", "item_id": "c", "rating": 2, "timestamp": 1},
        {"user_id": "v", "item_id": "b", "rating": 5, "timestamp": 2},
        {"user_id": "v", "item_id": "a", "rating": 4, "timestamp": 3},
    ]
    candidates = build_candidate_set("u", "snap-1", rows, 10)
    result = ranker_type(seed=7).fit(rows, 10).rank(candidates, k=2)

    assert result.candidate_item_ids == ("a", "b", "c")
    assert [(item.item_id, item.rank) for item in result.items] == [("a", 1), ("b", 2)]
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("timestamp", ["nan", "inf", True])
def test_candidate_contract_rejects_invalid_time(timestamp) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        CandidateSet("u", "snap-1", timestamp, ())
