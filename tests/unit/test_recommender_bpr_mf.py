import pytest

from trustrec.recommenders.bpr_mf import BPRMFRanker, rank_bpr_mf
from trustrec.recommenders.contracts import CandidateSet, build_candidate_set


def _training_rows() -> list[dict[str, object]]:
    return [
        {"user_id": "u1", "item_id": "h", "rating": 5, "timestamp": 1},
        {"user_id": "u2", "item_id": "h", "rating": 5, "timestamp": 2},
        {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 3},
        {"user_id": "u3", "item_id": "a", "rating": 5, "timestamp": 4},
        {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 5},
        {"user_id": "u4", "item_id": "b", "rating": 5, "timestamp": 6},
    ]


def test_bpr_mf_is_seed_deterministic_and_returns_mf_component() -> None:
    rows = _training_rows()
    candidates = CandidateSet("u1", "snap-1", 10, ("a", "b"), frozenset({"h"}))

    first = rank_bpr_mf(rows, candidates, seed=7, epochs=25, embedding_dimensions=8)
    second = rank_bpr_mf(
        list(reversed(rows)), candidates, seed=7, epochs=25, embedding_dimensions=8
    )

    assert first.to_dict() == second.to_dict()
    assert first.seed == 7
    assert first.model_id == "b2_bpr_mf"
    assert set(first.items[0].component_scores) >= {"mf"}
    assert all(
        item.total_score == pytest.approx(item.component_scores["mf"]) for item in first.items
    )


def test_bpr_mf_falls_back_to_popularity_for_a_user_without_positive_history() -> None:
    result = rank_bpr_mf(
        [
            {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 2},
        ],
        CandidateSet("u1", "snap-1", 10, ("a", "b")),
        seed=7,
        embedding_dimensions=4,
        epochs=2,
    )

    assert result.fallback_reason == "no_positive_history"
    assert [item.item_id for item in result.items] == ["a", "b"]
    assert all(item.component_scores["mf"] == 0.0 for item in result.items)


def test_bpr_mf_uses_only_first_pre_cutoff_events() -> None:
    rows = [
        {"review_id": "later", "user_id": "v", "item_id": "a", "rating": 5, "timestamp": 2},
        {"review_id": "first", "user_id": "v", "item_id": "a", "rating": 2, "timestamp": 1},
        {"user_id": "w", "item_id": "b", "rating": 5, "timestamp": 3},
        {"user_id": "new", "item_id": "future", "rating": 5, "timestamp": 11},
    ]
    candidates = build_candidate_set("u", "snap-1", rows, 10)
    result = BPRMFRanker(seed=7, epochs=3, embedding_dimensions=4).fit(rows, 10).rank(candidates)

    assert result.candidate_item_ids == ("a", "b")
    assert result.model_hash
    assert "future" not in result.model_hash


def test_bpr_mf_rejects_invalid_training_configuration() -> None:
    with pytest.raises(ValueError, match="embedding_dimensions"):
        BPRMFRanker(embedding_dimensions=0)
    with pytest.raises(ValueError, match="learning_rate"):
        BPRMFRanker(learning_rate=0)
    with pytest.raises(ValueError, match="regularization"):
        BPRMFRanker(regularization=-1)
