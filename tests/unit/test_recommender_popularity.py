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


DAY_MS = 86_400_000
CUTOFF_MS = 100 * DAY_MS


def _recent_rows() -> list[dict[str, object]]:
    def row(user: str, item: str, day: float, rating: int = 5) -> dict[str, object]:
        return {"user_id": user, "item_id": item, "rating": rating, "timestamp": day * DAY_MS}

    return [
        row("u1", "old", 1),
        row("u2", "old", 2),
        row("u3", "old", 3),
        row("u4", "new", 95),
        row("u5", "new", 96),
        row("u6", "mid", 50),
        row("u7", "mid", 97, rating=2),
        row("u8", "tie", 98),
        row("u9", "late", 100),
    ]


def test_recent_popularity_counts_positive_events_in_the_window_only() -> None:
    from trustrec.recommenders.popularity import RecentPopularityRanker

    candidates = CandidateSet("u0", "snap-1", CUTOFF_MS, ("mid", "new", "old", "tie"))
    ranker = RecentPopularityRanker(window_days=10).fit(_recent_rows(), CUTOFF_MS)

    result = ranker.rank(candidates)

    assert [item.item_id for item in result.items] == ["new", "tie", "old", "mid"]
    assert [item.total_score for item in result.items] == [2.0, 1.0, 0.0, 0.0]
    wide = RecentPopularityRanker(window_days=1000).fit(_recent_rows(), CUTOFF_MS)
    assert [item.item_id for item in wide.rank(candidates).items][:1] == ["old"]
    assert ranker.model_hash != wide.model_hash


def test_recent_popularity_matches_between_row_and_table_inputs() -> None:
    import pandas as pd

    from trustrec.recommenders.matrix import build_interaction_matrix

    rows = _recent_rows()
    from_rows = build_interaction_matrix(rows, CUTOFF_MS)
    from_frame = build_interaction_matrix(pd.DataFrame(rows), CUTOFF_MS)

    assert from_rows.fingerprint == from_frame.fingerprint
    assert from_rows.recent_popularity(10).tolist() == from_frame.recent_popularity(10).tolist()
    with pytest.raises(ValueError, match="window_days"):
        from_rows.recent_popularity(0)


def test_recent_popularity_round_trips_through_the_registry(tmp_path) -> None:
    from trustrec.recommenders.matrix import build_interaction_matrix
    from trustrec.recommenders.registry import BaselineParameters, create_baseline, load_baseline

    matrix = build_interaction_matrix(_recent_rows(), CUTOFF_MS)
    ranker = create_baseline("b5_recent_popularity", BaselineParameters(window_days=10))
    ranker.fit(None, CUTOFF_MS, "snap-1", matrix=matrix)
    ranker.save(tmp_path / "b5.npz")

    restored = load_baseline(tmp_path / "b5.npz", matrix)

    assert restored.model_hash == ranker.model_hash
    assert restored.window_days == 10.0
    with pytest.raises(ValueError, match="window_days"):
        create_baseline("b5_recent_popularity", BaselineParameters(window_days=-1))
