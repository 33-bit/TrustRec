"""Personalized PageRank on a positive user-item interaction graph."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from scipy import sparse

from trustrec.recommenders.base import MatrixRanker
from trustrec.recommenders.contracts import CandidateSet, RankingResult
from trustrec.recommenders.matrix import InteractionMatrix


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class PersonalizedPageRankRanker(MatrixRanker):
    """Rank candidates by PageRank mass reached from one user node.

    The graph has one node per user and item with at least one positive edge.
    Power iteration runs as sparse matrix products, several restart users at a
    time, and each user's column stops when it meets ``tol``.
    """

    model_id = "b3_ppr"
    score_component = "ppr"
    zero_score_fallback = "no_graph_path"

    def __init__(
        self,
        damping: float = 0.85,
        positive_threshold: float = 4.0,
        max_iter: int = 100,
        tol: float = 1e-12,
        seed: int | None = None,
        batch_size: int = 64,
    ) -> None:
        if not math.isfinite(float(damping)) or not 0 <= damping < 1:
            raise ValueError("damping must be finite and in [0, 1)")
        if not math.isfinite(float(tol)) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        super().__init__(positive_threshold=positive_threshold, seed=seed)
        self.damping = float(damping)
        self.max_iter = _positive_integer("max_iter", max_iter)
        self.tol = float(tol)
        self.batch_size = _positive_integer("batch_size", batch_size)
        self._score_cache: dict[str, np.ndarray] = {}

    def _configuration(self) -> dict[str, Any]:
        return {
            "damping": self.damping,
            "positive_threshold": self.positive_threshold,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "cutoff_rule": "timestamp_strictly_before",
            "edge_rule": "positive_ratings_only",
            "dangling_rule": "restart_mass_to_user",
            "tie_break": "item_id_ascending",
        }

    def _hash_configuration(self) -> dict[str, Any]:
        return {
            "damping": self.damping,
            "positive_threshold": self.positive_threshold,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def _fit_matrix(self, matrix: InteractionMatrix) -> None:
        positive = matrix.positive
        user_degree = np.diff(positive.indptr).astype(np.float64)
        item_degree = matrix.popularity.astype(np.float64)
        inverse_user = np.divide(
            1.0, user_degree, out=np.zeros_like(user_degree), where=user_degree > 0
        )
        inverse_item = np.divide(
            1.0, item_degree, out=np.zeros_like(item_degree), where=item_degree > 0
        )
        # Walk steps as pre-scaled sparse matrices:
        # user -> item is B^T D_u^-1 and item -> user is B D_i^-1.
        self._item_from_user = (positive.T @ sparse.diags(inverse_user)).tocsr()
        self._user_from_item = (positive @ sparse.diags(inverse_item)).tocsr()
        # Users with one positive edge only send an item's mass back to itself,
        # so their two-step contribution is a diagonal item term.
        multi = np.flatnonzero(user_degree > 1)
        single = user_degree == 1
        self._item_from_multi = self._item_from_user[:, multi].tocsr()
        self._multi_from_item = self._user_from_item[multi].tocsr()
        single_counts = np.asarray(positive[single].sum(axis=0)).ravel()
        self._self_return = single_counts * inverse_item
        self._score_cache = {}

    def item_scores(self, user_ids: Sequence[str]) -> np.ndarray:
        """Return PageRank item scores with one column per user.

        Every user must have at least one positive edge. The power iteration
        ``p_u' = d B D_i^-1 p_i + (1 - d) e_r`` and ``p_i' = d B^T D_u^-1 p_u``
        starts at ``p_u = e_r``. Item mass therefore follows
        ``p_i(t) = d^2 M p_i(t - 2) + d (1 - d) v_r`` with ``M = B^T D_u^-1 B D_i^-1``
        and ``v_r = B^T D_u^-1 e_r``, which this method iterates without
        storing user mass. Every graph node has an edge, so no dangling mass
        exists. A column stops at the first step whose change over all user
        and item nodes is at most ``tol``.
        """

        matrix = self.matrix
        restart_users = np.array([matrix.user_index[user_id] for user_id in user_ids])
        if restart_users.size and (np.diff(matrix.positive.indptr)[restart_users] == 0).any():
            raise ValueError("every user must have a positive edge in the graph")
        damping = self.damping
        restart = self._item_from_user[:, restart_users].toarray()
        result = np.zeros((matrix.n_items, restart_users.size))
        active = np.arange(restart_users.size)
        previous = np.zeros_like(restart)
        current = damping * restart
        # Step 1 moves user mass by ``damping`` at the restart node.
        if damping <= self.tol:
            result[:, active] = current
            return result
        for _ in range(2, self.max_iter + 1):
            following = (
                damping
                * damping
                * (
                    self._item_from_multi @ (self._multi_from_item @ previous)
                    + self._self_return[:, None] * previous
                )
                + damping * (1.0 - damping) * restart
            )
            done = np.abs(following - current).max(axis=0) <= self.tol
            if done.any():
                user_change = damping * np.abs(
                    self._user_from_item @ (current[:, done] - previous[:, done])
                ).max(axis=0)
                done[np.flatnonzero(done)[user_change > self.tol]] = False
            if done.any():
                result[:, active[done]] = following[:, done]
                keep = ~done
                active = active[keep]
                restart = restart[:, keep]
                current = current[:, keep]
                following = following[:, keep]
                if active.size == 0:
                    return result
            previous, current = current, following
        result[:, active] = current
        return result

    def prepare(self, user_ids: Sequence[str]) -> None:
        """Compute and cache scores for users with positive edges, in batches."""

        matrix = self.matrix
        graph_users = [
            user_id
            for user_id in user_ids
            if user_id in matrix.user_index
            and matrix.positive_indices(user_id).size > 0
            and user_id not in self._score_cache
        ]
        for start in range(0, len(graph_users), self.batch_size):
            chunk = graph_users[start : start + self.batch_size]
            scores = self.item_scores(chunk)
            for column, user_id in enumerate(chunk):
                self._score_cache[user_id] = scores[:, column].copy()

    def clear_cache(self) -> None:
        self._score_cache = {}

    def _user_scores(self, user_id: str, positive_items: np.ndarray) -> np.ndarray:
        cached = self._score_cache.get(user_id)
        if cached is not None:
            return cached
        return self.item_scores([user_id])[:, 0]


def rank_personalized_pagerank(
    interactions: Iterable[Mapping[str, Any]] | Any,
    candidate_set: CandidateSet,
    *,
    damping: float = 0.85,
    positive_threshold: float = 4.0,
    max_iter: int = 100,
    tol: float = 1e-12,
    snapshot_id: str | None = None,
    seed: int | None = None,
    k: int | None = None,
) -> RankingResult:
    """Fit and apply the personalized PageRank baseline in one call."""

    return (
        PersonalizedPageRankRanker(
            damping=damping,
            positive_threshold=positive_threshold,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        .fit(interactions, candidate_set.cutoff_timestamp, snapshot_id=snapshot_id)
        .rank(candidate_set, k=k)
    )
