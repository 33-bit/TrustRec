"""Compatibility API for evidence profile aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from trustrec.recommenders.contracts import Timestamp

from .evidence import (
    AggregatedEvidence,
    aggregate_evidence,
    build_user_aspect_weights,
    compute_aspect_priors,
)


@dataclass(frozen=True)
class EvidenceAggregator:
    """Reusable evidence aggregator with fixed snapshot configuration."""

    shrinkage_lambda: float = 5.0
    recency_half_life_days: float = 0.0
    duplicate_penalty: str | float = "inverse_group_size"

    def aggregate(
        self,
        rows: Iterable[Mapping[str, Any] | Any],
        *,
        cutoff_timestamp: Timestamp,
        prior_scores: Mapping[Any, Any] | None = None,
        items: Iterable[str] | None = None,
        aspects: Iterable[str] | None = None,
    ) -> dict[tuple[str, str], AggregatedEvidence]:
        return aggregate_evidence(
            rows,
            cutoff_timestamp=cutoff_timestamp,
            prior_scores=prior_scores,
            shrinkage_lambda=self.shrinkage_lambda,
            recency_half_life_days=self.recency_half_life_days,
            duplicate_penalty=self.duplicate_penalty,
            items=items,
            aspects=aspects,
        )

    def priors(
        self,
        rows: Iterable[Mapping[str, Any] | Any],
        *,
        cutoff_timestamp: Timestamp,
        aspects: Iterable[str] | None = None,
    ) -> dict[str, float]:
        return compute_aspect_priors(
            rows,
            cutoff_timestamp=cutoff_timestamp,
            aspects=aspects,
            recency_half_life_days=self.recency_half_life_days,
            duplicate_penalty=self.duplicate_penalty,
        )

    def user_weights(
        self,
        rows: Iterable[Mapping[str, Any] | Any],
        *,
        user_id: str,
        aspects: Iterable[str],
        alpha: float = 1.0,
        prior_distribution: Mapping[str, float] | None = None,
        cutoff_timestamp: Timestamp | None = None,
    ) -> dict[str, float]:
        return build_user_aspect_weights(
            rows,
            user_id=user_id,
            aspects=aspects,
            alpha=alpha,
            prior_distribution=prior_distribution,
            cutoff_timestamp=cutoff_timestamp,
        )


__all__ = [
    "AggregatedEvidence",
    "EvidenceAggregator",
    "aggregate_evidence",
    "build_user_aspect_weights",
    "compute_aspect_priors",
]
