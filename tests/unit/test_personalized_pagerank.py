import pytest

from trustrec.graph.personalized_pagerank import (
    PersonalizedPageRankRanker,
    rank_personalized_pagerank,
)
from trustrec.recommenders.contracts import CandidateSet, build_candidate_set


def test_personalized_pagerank_follows_positive_bipartite_paths() -> None:
    result = rank_personalized_pagerank(
        [
            {"user_id": "u1", "item_id": "h", "rating": 5, "timestamp": 1},
            {"user_id": "u2", "item_id": "h", "rating": 5, "timestamp": 2},
            {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 3},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 4},
        ],
        CandidateSet("u1", "snap-1", 10, ("a", "b"), frozenset({"h"})),
        damping=0.85,
    )

    assert [item.item_id for item in result.items] == ["a", "b"]
    assert result.items[0].component_scores["ppr"] > 0.0
    assert result.items[1].component_scores["ppr"] == pytest.approx(0.0)
    assert result.fallback_reason is None


def test_personalized_pagerank_falls_back_for_a_user_without_positive_edges() -> None:
    result = rank_personalized_pagerank(
        [
            {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 2},
        ],
        CandidateSet("u1", "snap-1", 10, ("a", "b")),
    )

    assert result.fallback_reason == "no_positive_history"
    assert [item.item_id for item in result.items] == ["a", "b"]
    assert all(item.component_scores["ppr"] == 0.0 for item in result.items)


def test_personalized_pagerank_reports_no_graph_path_and_ignores_future_rows() -> None:
    rows = [
        {"user_id": "u1", "item_id": "h", "rating": 5, "timestamp": 1},
        {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 2},
        {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 3},
        {"user_id": "u1", "item_id": "b", "rating": 5, "timestamp": 11},
    ]
    candidates = build_candidate_set("u1", "snap-1", rows, 10)
    before = PersonalizedPageRankRanker(seed=7).fit(rows[:3], 10).rank(candidates)
    after = PersonalizedPageRankRanker(seed=7).fit(rows, 10).rank(candidates)

    assert before.to_dict() == after.to_dict()
    assert before.fallback_reason == "no_graph_path"
    assert all(item.component_scores["ppr"] == 0.0 for item in before.items)


def test_personalized_pagerank_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="damping"):
        PersonalizedPageRankRanker(damping=1.0)
    with pytest.raises(ValueError, match="max_iter"):
        PersonalizedPageRankRanker(max_iter=0)
