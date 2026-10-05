"""Fixed-weight hybrid recommender (H0)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .hybrid import HybridScore, fixed_hybrid_details, fixed_hybrid_scores, rank_scores


def rank_fixed_hybrid(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
    *,
    mf_weight: float = 0.5,
    graph_weight: float = 0.25,
    aspect_weight: float = 0.25,
    k: int | None = None,
) -> tuple[HybridScore, ...]:
    """Rank a shared candidate set with fixed hybrid weights."""

    details = fixed_hybrid_details(
        mf_scores,
        graph_scores,
        aspect_scores,
        mf_weight=mf_weight,
        graph_weight=graph_weight,
        aspect_weight=aspect_weight,
    )
    return tuple(
        details[item_id]
        for item_id, _ in rank_scores(
            {key: value.total_score for key, value in details.items()}, k=k
        )
    )


@dataclass(frozen=True)
class FixedHybridRanker:
    """Reusable H0 scorer with validation-time weights."""

    mf_weight: float = 0.5
    graph_weight: float = 0.25
    aspect_weight: float = 0.25

    def score(
        self,
        mf_scores: Mapping[str, float],
        graph_scores: Mapping[str, float],
        aspect_scores: Mapping[str, float],
    ) -> dict[str, float]:
        return fixed_hybrid_scores(
            mf_scores,
            graph_scores,
            aspect_scores,
            mf_weight=self.mf_weight,
            graph_weight=self.graph_weight,
            aspect_weight=self.aspect_weight,
        )

    def rank(
        self,
        mf_scores: Mapping[str, float],
        graph_scores: Mapping[str, float],
        aspect_scores: Mapping[str, float],
        *,
        k: int | None = None,
    ) -> tuple[HybridScore, ...]:
        return rank_fixed_hybrid(
            mf_scores,
            graph_scores,
            aspect_scores,
            mf_weight=self.mf_weight,
            graph_weight=self.graph_weight,
            aspect_weight=self.aspect_weight,
            k=k,
        )


FixedHybrid = FixedHybridRanker
