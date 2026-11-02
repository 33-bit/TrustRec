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


def signature_rows(dataset: Any, *, signature: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "user_id": case.user_id,
            "snapshot_id": dataset.snapshot_id,
            "cutoff_timestamp": dataset.cutoff_timestamp,
            "candidate_rule": "catalog_minus_history",
            "candidate_count": case.candidate_count,
            "candidate_signature": signature or case.candidate_signature,
            "ranked_item_ids": ["a"],
        }
        for case in dataset.cases
    ]


def test_compare_model_rankings_accepts_candidate_signatures() -> None:
    dataset = fixture_dataset()
    results = compare_model_rankings(
        dataset,
        {"b0": fixture_rows(dataset), "b1": signature_rows(dataset)},
        k_values=(1,),
        bootstrap_samples=20,
    )

    assert results["b1"].metrics["ndcg@1"] == 1.0
    assert results["b1"].metrics["candidate_retention"] == 0.5


@pytest.mark.parametrize("field", ["candidate_signature", "candidate_count", "missing"])
def test_compare_model_rankings_rejects_wrong_candidate_declarations(field: str) -> None:
    dataset = fixture_dataset()
    rows = signature_rows(dataset)
    if field == "candidate_signature":
        rows[0]["candidate_signature"] = "0" * 64
    elif field == "candidate_count":
        rows[0]["candidate_count"] += 1
    else:
        del rows[0]["candidate_signature"]

    error = ValueError if field == "missing" else LeakageError
    with pytest.raises(error, match="candidate"):
        compare_model_rankings(dataset, {"b1": rows}, k_values=(1,), bootstrap_samples=20)


def test_compare_model_rankings_rejects_ranked_history_items() -> None:
    dataset = fixture_dataset()
    rows = signature_rows(dataset)
    rows[0]["ranked_item_ids"] = ["h", "a"]

    with pytest.raises(LeakageError, match="outside its candidate set"):
        compare_model_rankings(dataset, {"b1": rows}, k_values=(1,), bootstrap_samples=20)


def test_evaluation_cases_share_one_catalog_without_copies() -> None:
    interactions = [
        {"review_id": f"r{i}", "user_id": f"u{i}", "item_id": f"i{i}", "rating": 5, "timestamp": 1}
        for i in range(300)
    ]
    targets = [
        {"review_id": f"t{i}", "user_id": f"u{i}", "item_id": "i0", "rating": 5, "timestamp": 10}
        for i in range(1, 300)
    ]
    dataset = build_evaluation_cases(interactions, targets, snapshot_id="s", cutoff_timestamp=10)

    assert len(dataset.cases) == 299
    assert len({id(case.catalog_item_ids) for case in dataset.cases}) == 1
    assert all(case.candidate_count == 299 for case in dataset.cases)
    rows = signature_rows(dataset)
    for row in rows:
        row["ranked_item_ids"] = ["i0"]
    results = compare_model_rankings(dataset, {"b1": rows}, k_values=(1,), bootstrap_samples=50)

    assert results["b1"].metrics["recall@1"] == 1.0
