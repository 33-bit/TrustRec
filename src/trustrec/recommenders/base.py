"""Shared fit, validation, fallback, and top-K logic for matrix-backed rankers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .contracts import (
    CandidateSet,
    RankedItem,
    RankingResult,
    Timestamp,
    stable_model_hash,
    timestamp_value,
)
from .matrix import InteractionMatrix, build_interaction_matrix, top_k_order


def validate_k(k: int | None) -> None:
    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
        raise ValueError("k must be a positive integer or None")


@dataclass(frozen=True)
class TopKRecommendation:
    """Top-K items over the protocol candidate pool for one user."""

    user_id: str
    item_ids: tuple[str, ...]
    scores: tuple[float, ...]
    candidate_count: int
    candidate_signature: str
    fallback_reason: str | None


@dataclass(frozen=True)
class _Ordering:
    positions: np.ndarray
    totals: np.ndarray
    components: dict[str, np.ndarray]
    fallback_reason: str | None


class MatrixRanker:
    """Base class for baselines fitted on one ``InteractionMatrix``.

    Subclasses provide ``_fit_matrix`` and ``_user_scores``. Scores cover the
    whole catalog, so ``rank`` (one explicit candidate set) and
    ``recommend`` (catalog minus history, top-K) share one ordering rule.
    """

    model_id: ClassVar[str]
    score_component: ClassVar[str]
    requires_positive_history: ClassVar[bool] = True
    zero_score_fallback: ClassVar[str | None] = None
    popularity_tie_break: ClassVar[bool] = False
    include_popularity_component: ClassVar[bool] = False

    def __init__(self, positive_threshold: float = 4.0, seed: int | None = None) -> None:
        if not math.isfinite(float(positive_threshold)):
            raise ValueError("positive_threshold must be finite")
        self.positive_threshold = float(positive_threshold)
        self.seed = seed
        self._matrix: InteractionMatrix | None = None
        self._cutoff_timestamp: Timestamp | None = None
        self._snapshot_id: str | None = None
        self._model_hash: str | None = None

    # Subclass hooks -----------------------------------------------------

    def _configuration(self) -> dict[str, Any]:
        raise NotImplementedError

    def _hash_configuration(self) -> dict[str, Any]:
        return {"positive_threshold": self.positive_threshold}

    def _fit_matrix(self, matrix: InteractionMatrix) -> None:
        """Learn model state from the fitted matrix."""

    def _learned_state_hash(self) -> str | None:
        return None

    def _user_scores(self, user_id: str, positive_items: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def configuration(self) -> dict[str, Any]:
        """Return the configuration recorded in ranking results."""

        return self._configuration()

    # Fit ----------------------------------------------------------------

    def fit(
        self,
        interactions: Iterable[Mapping[str, Any]] | Any,
        cutoff_timestamp: Timestamp,
        snapshot_id: str | None = None,
        *,
        matrix: InteractionMatrix | None = None,
    ) -> MatrixRanker:
        """Fit on first events strictly before the cutoff.

        Pass a prebuilt ``matrix`` to share one snapshot load across models.
        """

        if matrix is None:
            matrix = build_interaction_matrix(
                interactions, cutoff_timestamp, positive_threshold=self.positive_threshold
            )
        else:
            if timestamp_value(matrix.cutoff_timestamp) != timestamp_value(cutoff_timestamp):
                raise ValueError("matrix cutoff does not match the requested cutoff")
            if matrix.positive_threshold != self.positive_threshold:
                raise ValueError("matrix positive_threshold does not match the ranker")
        self._matrix = matrix
        self._cutoff_timestamp = cutoff_timestamp
        self._snapshot_id = snapshot_id
        self._fit_matrix(matrix)
        self._model_hash = stable_model_hash(
            {
                "model_id": self.model_id,
                **self._hash_configuration(),
                "seed": self.seed,
                "cutoff_timestamp": timestamp_value(cutoff_timestamp),
                "snapshot_id": snapshot_id,
                "data_fingerprint": matrix.fingerprint,
                "learned_state_sha256": self._learned_state_hash(),
            }
        )
        return self

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before reading model_hash")
        return self._model_hash

    @property
    def matrix(self) -> InteractionMatrix:
        if self._matrix is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before ranking")
        return self._matrix

    # Ranking ------------------------------------------------------------

    def rank(self, candidate_set: CandidateSet, k: int | None = None) -> RankingResult:
        """Rank the exact candidates from a shared candidate set."""

        self._require_fitted(candidate_set)
        validate_k(k)
        self._validate_candidate_set(candidate_set)
        matrix = self.matrix
        candidates = np.array(
            sorted(matrix.item_index[item_id] for item_id in candidate_set.candidate_item_ids),
            dtype=np.int64,
        )
        positives = matrix.positive_indices(candidate_set.user_id)
        if candidate_set.history_item_ids:
            declared = [
                matrix.item_index[item_id]
                for item_id in candidate_set.history_item_ids
                if item_id in matrix.item_index
            ]
            positives = np.intersect1d(positives, np.array(declared, dtype=np.int64))
        ordering = self._order(candidate_set.user_id, positives, candidates, k)
        items = tuple(
            RankedItem(
                item_id=matrix.item_ids[int(candidates[position])],
                rank=rank,
                total_score=float(ordering.totals[position]),
                component_scores={
                    name: float(values[position]) for name, values in ordering.components.items()
                },
            )
            for rank, position in enumerate(ordering.positions, start=1)
        )
        return RankingResult(
            model_id=self.model_id,
            user_id=candidate_set.user_id,
            snapshot_id=candidate_set.snapshot_id,
            cutoff_timestamp=candidate_set.cutoff_timestamp,
            candidate_item_ids=candidate_set.candidate_item_ids,
            items=items,
            configuration=self._configuration(),
            seed=self.seed,
            model_hash=self._model_hash or "",
            fallback_reason=ordering.fallback_reason,
        )

    def recommend(self, user_id: str, k: int) -> TopKRecommendation:
        """Return the top-K items from the catalog minus the user's history."""

        if self._model_hash is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before ranking")
        validate_k(k)
        if k is None:
            raise ValueError("k must be a positive integer")
        matrix = self.matrix
        candidates = matrix.candidate_indices(user_id)
        ordering = self._order(user_id, matrix.positive_indices(user_id), candidates, k)
        return TopKRecommendation(
            user_id=user_id,
            item_ids=tuple(matrix.item_ids[int(candidates[p])] for p in ordering.positions),
            scores=tuple(float(ordering.totals[p]) for p in ordering.positions),
            candidate_count=int(candidates.size),
            candidate_signature=matrix.candidate_signature(user_id),
            fallback_reason=ordering.fallback_reason,
        )

    def _order(
        self,
        user_id: str,
        positive_items: np.ndarray,
        candidates: np.ndarray,
        k: int | None,
    ) -> _Ordering:
        if candidates.size == 0:
            empty = np.empty(0)
            return _Ordering(np.empty(0, dtype=np.int64), empty, {}, "no_candidates")
        popularity = self.matrix.popularity[candidates].astype(np.float64)
        if self.requires_positive_history and positive_items.size == 0:
            return self._popularity_ordering(candidates, popularity, k, "no_positive_history")
        scores = self._user_scores(user_id, positive_items)[candidates]
        if self.zero_score_fallback is not None and not (scores > 0.0).any():
            return self._popularity_ordering(candidates, popularity, k, self.zero_score_fallback)
        positions = top_k_order(
            scores,
            candidates,
            k,
            secondary=popularity if self.popularity_tie_break else None,
        )
        components = {self.score_component: scores}
        if self.include_popularity_component:
            components["popularity"] = popularity
        return _Ordering(positions, scores, components, None)

    def _popularity_ordering(
        self,
        candidates: np.ndarray,
        popularity: np.ndarray,
        k: int | None,
        reason: str,
    ) -> _Ordering:
        positions = top_k_order(popularity, candidates, k)
        components = {
            self.score_component: np.zeros(candidates.size),
            "popularity": popularity,
        }
        return _Ordering(positions, popularity, components, reason)

    # Validation ---------------------------------------------------------

    def _validate_candidate_set(self, candidate_set: CandidateSet) -> None:
        if self._snapshot_id is not None and candidate_set.snapshot_id != self._snapshot_id:
            raise ValueError("candidate snapshot does not match fitted snapshot")
        matrix = self.matrix
        history = {matrix.item_ids[i] for i in matrix.history_indices(candidate_set.user_id)}
        if set(candidate_set.candidate_item_ids) & history:
            raise ValueError("candidate_item_ids must exclude history items")
        if any(item_id not in matrix.item_index for item_id in candidate_set.candidate_item_ids):
            raise ValueError("candidate_item_ids must be known before the cutoff")

    def _require_fitted(self, candidate_set: CandidateSet) -> None:
        if self._cutoff_timestamp is None or self._model_hash is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before ranking")
        if timestamp_value(candidate_set.cutoff_timestamp) != timestamp_value(
            self._cutoff_timestamp
        ):
            raise ValueError("candidate cutoff does not match fitted cutoff")
