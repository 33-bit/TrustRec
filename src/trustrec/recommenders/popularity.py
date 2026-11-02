"""Most-popular ranking baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .base import MatrixRanker
from .contracts import CandidateSet, RankingResult


class PopularityRanker(MatrixRanker):
    """Rank candidates by positive interactions before one cutoff."""

    model_id = "b0_most_popular"
    score_component = "popularity"
    requires_positive_history = False

    def _configuration(self) -> dict[str, Any]:
        return {
            "positive_threshold": self.positive_threshold,
            "cutoff_rule": "timestamp_strictly_before",
            "tie_break": "item_id_ascending",
        }

    def _user_scores(self, user_id: str, positive_items: np.ndarray) -> np.ndarray:
        return self.matrix.popularity.astype(np.float64)


def rank_popularity(
    interactions: Iterable[Mapping[str, Any]] | Any,
    candidate_set: CandidateSet,
    *,
    positive_threshold: float = 4.0,
    snapshot_id: str | None = None,
    seed: int | None = None,
    k: int | None = None,
) -> RankingResult:
    """Fit and apply the most-popular baseline in one call."""

    return (
        PopularityRanker(positive_threshold=positive_threshold, seed=seed)
        .fit(interactions, candidate_set.cutoff_timestamp, snapshot_id=snapshot_id)
        .rank(candidate_set, k=k)
    )
