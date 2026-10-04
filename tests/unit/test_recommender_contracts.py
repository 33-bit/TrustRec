import pytest

from trustrec.recommenders.contracts import (
    CandidateSet,
    RankedItem,
    RankingResult,
    build_candidate_set,
    first_interactions_before_cutoff,
)


def test_build_candidate_set_removes_all_pre_cutoff_history_items() -> None:
    result = build_candidate_set(
        "u1",
        "snap-1",
        [
            {"user_id": "u1", "item_id": "seen-low", "rating": 2, "timestamp": 1},
            {"user_id": "u1", "item_id": "seen-high", "rating": 5, "timestamp": 2},
            {"user_id": "u2", "item_id": "candidate", "rating": 5, "timestamp": 3},
            {"user_id": "u1", "item_id": "future", "rating": 5, "timestamp": 20},
        ],
        cutoff_timestamp=10,
    )

    assert result.candidate_item_ids == ("candidate",)
    assert result.history_item_ids == frozenset({"seen-low", "seen-high"})


def test_candidate_set_rejects_history_overlap() -> None:
    with pytest.raises(ValueError, match="exclude history"):
        CandidateSet(
            user_id="u1",
            snapshot_id="snap-1",
            cutoff_timestamp=10,
            candidate_item_ids=("i1",),
            history_item_ids=frozenset({"i1"}),
        )


def test_build_candidate_set_rejects_unknown_explicit_candidate() -> None:
    with pytest.raises(ValueError, match="known before the cutoff"):
        build_candidate_set(
            "u1",
            "snap-1",
            [{"user_id": "u2", "item_id": "known", "timestamp": 1}],
            cutoff_timestamp=10,
            candidate_item_ids=("unknown",),
        )


def test_first_event_tie_break_is_independent_of_input_order() -> None:
    rows = [
        {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 1},
        {"user_id": "u1", "item_id": "a", "rating": 2, "timestamp": 1},
        {"user_id": "u2", "item_id": "b", "rating": 5, "timestamp": 2},
    ]

    first = first_interactions_before_cutoff(rows, 10)
    second = first_interactions_before_cutoff(list(reversed(rows)), 10)

    assert first == second
    assert first[0]["rating"] == 2


def test_ranking_result_keeps_ranked_items_and_candidate_ids() -> None:
    item = RankedItem("i1", 1, 2.0, {"popularity": 2.0})
    result = RankingResult(
        model_id="b0_most_popular",
        user_id="u1",
        snapshot_id="snap-1",
        cutoff_timestamp=10,
        candidate_item_ids=("i1",),
        items=(item,),
        configuration={"positive_threshold": 4.0},
        seed=None,
        model_hash="a" * 64,
    )

    assert result.items[0].component_scores["popularity"] == 2.0
