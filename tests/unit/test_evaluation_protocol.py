from __future__ import annotations

from typing import Any

import pytest

from trustrec.evaluation.leakage import LeakageError
from trustrec.evaluation.protocol import build_evaluation_cases, compare_model_rankings


def test_build_evaluation_cases_removes_history_and_future_targets() -> None:
    dataset = build_evaluation_cases(
        interactions=[
            {
                "review_id": "r1",
                "user_id": "u1",
                "item_id": "seen",
                "rating": 2,
                "timestamp": 1,
            },
            {
                "review_id": "r2",
                "user_id": "u2",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 2,
            },
        ],
        targets=[
            {
                "review_id": "r3",
                "user_id": "u1",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 10,
            },
            {
                "review_id": "r4",
                "user_id": "u1",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 9,
            },
        ],
        snapshot_id="snap-1",
        cutoff_timestamp=10,
    )

    assert dataset.cases[0].candidate_item_ids == ("candidate",)
    assert dataset.cases[0].target_item_ids == frozenset({"candidate"})
    assert dataset.exclusions["target_before_cutoff"] == 1


def test_build_evaluation_cases_records_unknown_and_non_positive_targets() -> None:
    dataset = build_evaluation_cases(
        interactions=[
            {"review_id": "r1", "user_id": "u1", "item_id": "seen", "rating": 5, "timestamp": 1},
            {
                "review_id": "r2",
                "user_id": "u2",
                "item_id": "candidate",
                "rating": 5,
                "timestamp": 2,
            },
        ],
        targets=[
            {
                "review_id": "r3",
                "user_id": "u1",
                "item_id": "unknown",
                "rating": 5,
                "timestamp": 10,
            },
            {"review_id": "r4", "user_id": "u1", "item_id": "seen", "rating": 5, "timestamp": 11},
            {
                "review_id": "r5",
                "user_id": "u1",
                "item_id": "candidate",
                "rating": 3,
                "timestamp": 12,
            },
        ],
        snapshot_id="snap-1",
        cutoff_timestamp=10,
    )

    assert dataset.cases == ()
    assert dataset.exclusions["target_unknown_item"] == 1
    assert dataset.exclusions["target_in_history"] == 1
    assert dataset.exclusions["target_not_positive"] == 1


def fixture_dataset() -> Any:
    return build_evaluation_cases(
        interactions=[
            {"review_id": "r1", "user_id": "u1", "item_id": "h", "rating": 5, "timestamp": 1},
            {"review_id": "r2", "user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 2},
            {"review_id": "r3", "user_id": "u2", "item_id": "b", "rating": 5, "timestamp": 3},
        ],
        targets=[
            {"review_id": "r4", "user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 10},
        ],
        snapshot_id="snap-1",
        cutoff_timestamp=10,
    )


def fixture_rows(
    dataset: Any, *, candidate_item_ids: list[str] | None = None, reverse: bool = False
) -> list[dict[str, Any]]:
    rows = []
    for case in dataset.cases:
        candidates = candidate_item_ids or list(case.candidate_item_ids)
        ranked = list(reversed(candidates)) if reverse else list(candidates)
        rows.append(
            {
                "user_id": case.user_id,
                "snapshot_id": dataset.snapshot_id,
                "cutoff_timestamp": dataset.cutoff_timestamp,
                "candidate_item_ids": candidates,
                "ranked_item_ids": ranked,
                "explanation_coverage": 1.0,
                "latency_ms": 1.0,
            }
        )
    return rows


def test_compare_model_rankings_rejects_candidate_set_mismatch() -> None:
    dataset = fixture_dataset()
    rows = {
        "b0": fixture_rows(dataset),
        "t0": fixture_rows(dataset, candidate_item_ids=["wrong"]),
    }

    with pytest.raises(LeakageError, match="candidate"):
        compare_model_rankings(
            dataset,
            rows,
            k_values=(5,),
            bootstrap_seed=7,
            bootstrap_samples=20,
            confidence_level=0.95,
        )


def test_compare_model_rankings_uses_shared_user_bootstrap() -> None:
    dataset = fixture_dataset()
    results = compare_model_rankings(
        dataset,
        {"b0": fixture_rows(dataset), "t0": fixture_rows(dataset, reverse=True)},
        k_values=(1,),
        bootstrap_seed=7,
        bootstrap_samples=100,
        confidence_level=0.95,
        reference_model="b0",
    )

    assert results["t0"].bootstrap["ndcg@1"].delta_estimate < 0
    assert results["b0"].counts["evaluated_users"] == 1
