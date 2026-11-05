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


class RecentPopularityRanker(MatrixRanker):
    """Rank candidates by positive interactions in a window that ends at the cutoff.

    Ties, including items with no recent positive events, break by all-time
    popularity and then by ascending item ID. Only events strictly before the
    cutoff count.
    """

    model_id = "b5_recent_popularity"
    score_component = "recent_popularity"
    requires_positive_history = False
    popularity_tie_break = True
    include_popularity_component = True

    def __init__(
        self,
        window_days: float = 365.0,
        positive_threshold: float = 4.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(positive_threshold=positive_threshold, seed=seed)
        if isinstance(window_days, bool) or not np.isfinite(float(window_days)):
            raise ValueError("window_days must be finite and positive")
        if float(window_days) <= 0:
            raise ValueError("window_days must be finite and positive")
        self.window_days = float(window_days)
        self._recent: np.ndarray | None = None

    def _configuration(self) -> dict[str, Any]:
        return {
            "positive_threshold": self.positive_threshold,
            "window_days": self.window_days,
            "cutoff_rule": "timestamp_strictly_before",
            "tie_break": "all_time_popularity_then_item_id_ascending",
        }

    def _hash_configuration(self) -> dict[str, Any]:
        return {"positive_threshold": self.positive_threshold, "window_days": self.window_days}

    def init_parameters(self) -> dict[str, Any]:
        return {**super().init_parameters(), "window_days": self.window_days}

    def _fit_matrix(self, matrix: Any) -> None:
        self._recent = matrix.recent_popularity(self.window_days).astype(np.float64)

    def _user_scores(self, user_id: str, positive_items: np.ndarray) -> np.ndarray:
        assert self._recent is not None
        return self._recent


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
