"""Item-based collaborative filtering baseline."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .base import MatrixRanker
from .contracts import CandidateSet, RankingResult
from .matrix import InteractionMatrix


def cosine_similarity(left_users: set[str], right_users: set[str]) -> float:
    """Return cosine similarity for two binary item user sets."""

    if not left_users or not right_users:
        return 0.0
    intersection = len(left_users & right_users)
    if intersection == 0:
        return 0.0
    return intersection / math.sqrt(len(left_users) * len(right_users))


class ItemKNNRanker(MatrixRanker):
    """Rank candidates by similarity to the user's positive item history.

    The score of a candidate is the sum of its strongest ``n_neighbors``
    binary cosine similarities to the user's positive history items. Scores
    come from one sparse co-occurrence product per user.
    """

    model_id = "b1_item_knn"
    score_component = "knn"
    zero_score_fallback = "no_similar_items"
    popularity_tie_break = True
    include_popularity_component = True

    def __init__(
        self,
        positive_threshold: float = 4.0,
        n_neighbors: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(positive_threshold=positive_threshold, seed=seed)
        if n_neighbors == 0:
            n_neighbors = None
        if n_neighbors is not None and (
            not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be positive, zero, or None")
        self.n_neighbors = n_neighbors
        self._item_users = None
        self._degrees = np.empty(0)

    def _configuration(self) -> dict[str, Any]:
        return {
            "positive_threshold": self.positive_threshold,
            "neighbor_limit": self.n_neighbors or 0,
            "cutoff_rule": "timestamp_strictly_before",
            "tie_break": "item_id_ascending",
        }

    def init_parameters(self) -> dict[str, Any]:
        return {
            "positive_threshold": self.positive_threshold,
            "n_neighbors": self.n_neighbors,
            "seed": self.seed,
        }

    def _hash_configuration(self) -> dict[str, Any]:
        return {"positive_threshold": self.positive_threshold, "n_neighbors": self.n_neighbors}

    def _fit_matrix(self, matrix: InteractionMatrix) -> None:
        self._item_users = matrix.positive.T.tocsr()
        self._degrees = matrix.popularity.astype(np.float64)

    def _user_scores(self, user_id: str, positive_items: np.ndarray) -> np.ndarray:
        matrix = self.matrix
        co_counts = (self._item_users[positive_items] @ matrix.positive).tocoo()
        rows = co_counts.row
        columns = co_counts.col.astype(np.int64)
        similarities = co_counts.data / np.sqrt(
            self._degrees[positive_items[rows]] * self._degrees[columns]
        )
        # Sort each candidate's similarities in descending order so the sum
        # and the neighbor cut do not depend on history order.
        scores = np.zeros(matrix.n_items)
        if columns.size == 0:
            return scores
        order = np.lexsort((-similarities, columns))
        columns = columns[order]
        similarities = similarities[order]
        starts = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
        if self.n_neighbors is not None:
            group_sizes = np.diff(np.r_[starts, columns.size])
            within_group = np.arange(columns.size) - np.repeat(starts, group_sizes)
            keep = within_group < self.n_neighbors
            columns = columns[keep]
            similarities = similarities[keep]
            starts = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
        # Extended precision keeps sums equal to correctly rounded ``math.fsum``
        # results in practice, so near-ties resolve as in the row baseline.
        sums = np.add.reduceat(similarities.astype(np.longdouble), starts)
        scores[columns[starts]] = sums.astype(np.float64)
        return scores


def rank_item_knn(
    interactions: Iterable[Mapping[str, Any]] | Any,
    candidate_set: CandidateSet,
    *,
    positive_threshold: float = 4.0,
    n_neighbors: int | None = None,
    snapshot_id: str | None = None,
    seed: int | None = None,
    k: int | None = None,
) -> RankingResult:
    """Fit and apply the item kNN baseline in one call."""

    return (
        ItemKNNRanker(
            positive_threshold=positive_threshold,
            n_neighbors=n_neighbors,
            seed=seed,
        )
        .fit(interactions, candidate_set.cutoff_timestamp, snapshot_id=snapshot_id)
        .rank(candidate_set, k=k)
    )
