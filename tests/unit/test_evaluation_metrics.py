import math

import pytest

from trustrec.evaluation.metrics import (
    intra_list_diversity,
    ndcg_at_k,
    paired_user_bootstrap,
    precision_at_k,
    recall_at_k,
)


def test_binary_ranking_metrics_use_top_k_and_target_count() -> None:
    ranked = ["b", "a", "c"]
    targets = {"a", "c"}

    assert recall_at_k(ranked, targets, 2) == 0.5
    assert precision_at_k(ranked, targets, 2) == 0.5
    assert ndcg_at_k(ranked, targets, 2) == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )


def test_metrics_reject_duplicate_ranked_items() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ndcg_at_k(["a", "a"], {"a"}, 2)


def test_metrics_return_zero_for_empty_targets() -> None:
    assert ndcg_at_k(["a"], set(), 1) == 0.0
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert precision_at_k(["a"], set(), 1) == 0.0


def test_intra_list_diversity_uses_pairwise_jaccard_distance() -> None:
    aspects = {"a": {"story", "graphics"}, "b": {"story"}, "c": {"value"}}

    assert intra_list_diversity(["a", "b", "c"], aspects, 3) == pytest.approx(
        ((1 - 1 / 2) + 1 + 1) / 3
    )


def test_intra_list_diversity_is_unavailable_without_aspects() -> None:
    assert intra_list_diversity(["a", "b"], {}, 2) is None


def test_paired_bootstrap_reports_reproducible_model_delta() -> None:
    values = {"b0": [0.0, 1.0, 1.0], "t0": [1.0, 1.0, 1.0]}

    result = paired_user_bootstrap(
        values,
        seed=7,
        samples=200,
        confidence_level=0.95,
        reference_model="b0",
    )
    repeated = paired_user_bootstrap(
        values,
        seed=7,
        samples=200,
        confidence_level=0.95,
        reference_model="b0",
    )

    assert result == repeated
    assert result["t0"].estimate == pytest.approx(1.0)
    assert result["t0"].delta_estimate == pytest.approx(1 / 3)
    assert result["t0"].lower <= result["t0"].estimate <= result["t0"].upper
