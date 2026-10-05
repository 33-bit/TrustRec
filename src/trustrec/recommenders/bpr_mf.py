"""Bayesian personalized ranking matrix-factorization baseline."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .contracts import (
    CandidateSet,
    RankedItem,
    RankingResult,
    Timestamp,
    first_interactions_before_cutoff,
    stable_model_hash,
    timestamp_value,
)
from .popularity import _rating_value, _validate_k


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class BPRMFRanker:
    """Rank candidates with seeded BPR matrix factorization."""

    model_id = "b2_bpr_mf"

    def __init__(
        self,
        embedding_dimensions: int = 64,
        learning_rate: float = 0.001,
        regularization: float = 0.0001,
        positive_threshold: float = 4.0,
        epochs: int = 50,
        negative_samples: int = 1,
        seed: int | None = 7,
    ) -> None:
        self.embedding_dimensions = _positive_integer("embedding_dimensions", embedding_dimensions)
        self.epochs = _positive_integer("epochs", epochs)
        self.negative_samples = _positive_integer("negative_samples", negative_samples)
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(regularization)) or regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        if not math.isfinite(float(positive_threshold)):
            raise ValueError("positive_threshold must be finite")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("seed must be an integer or None")
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.positive_threshold = float(positive_threshold)
        self.seed = int(seed) if seed is not None else None
        self._user_index: dict[str, int] = {}
        self._item_index: dict[str, int] = {}
        self._user_factors = np.empty((0, self.embedding_dimensions), dtype=np.float64)
        self._item_factors = np.empty((0, self.embedding_dimensions), dtype=np.float64)
        self._item_bias = np.empty(0, dtype=np.float64)
        self._positive_items_by_user: dict[str, frozenset[str]] = {}
        self._history_by_user: dict[str, frozenset[str]] = {}
        self._popularity: Counter[str] = Counter()
        self._catalog: frozenset[str] = frozenset()
        self._cutoff_timestamp: Timestamp | None = None
        self._snapshot_id: str | None = None
        self._model_hash: str | None = None

    def fit(
        self,
        interactions: Iterable[Mapping[str, Any]] | Any,
        cutoff_timestamp: Timestamp,
        snapshot_id: str | None = None,
    ) -> BPRMFRanker:
        """Fit on first positive events before the supplied cutoff."""

        rows = first_interactions_before_cutoff(interactions, cutoff_timestamp)
        catalog = tuple(sorted({row["item_id"] for row in rows}))
        history_by_user: dict[str, set[str]] = {}
        positive_items_by_user: dict[str, set[str]] = {}
        popularity: Counter[str] = Counter()
        positive_pairs: list[tuple[str, str]] = []
        for row in rows:
            user_id = row["user_id"]
            item_id = row["item_id"]
            history_by_user.setdefault(user_id, set()).add(item_id)
            rating = _rating_value(row)
            if rating is None or rating < self.positive_threshold:
                continue
            positive_items_by_user.setdefault(user_id, set()).add(item_id)
            popularity[item_id] += 1
            positive_pairs.append((user_id, item_id))

        positive_pairs = sorted(set(positive_pairs))
        users = tuple(sorted(positive_items_by_user))
        self._user_index = {user_id: index for index, user_id in enumerate(users)}
        self._item_index = {item_id: index for index, item_id in enumerate(catalog)}
        rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        self._user_factors = rng.normal(
            0.0,
            0.01,
            size=(len(users), self.embedding_dimensions),
        ).astype(np.float64)
        self._item_factors = rng.normal(
            0.0,
            0.01,
            size=(len(catalog), self.embedding_dimensions),
        ).astype(np.float64)
        self._item_bias = np.zeros(len(catalog), dtype=np.float64)

        negative_items_by_user = {
            user_id: tuple(
                item_id for item_id in catalog if item_id not in history_by_user[user_id]
            )
            for user_id in users
        }
        for _ in range(self.epochs):
            for pair_index in rng.permutation(len(positive_pairs)):
                user_id, positive_item_id = positive_pairs[int(pair_index)]
                negative_items = negative_items_by_user[user_id]
                if not negative_items:
                    continue
                user_index = self._user_index[user_id]
                positive_index = self._item_index[positive_item_id]
                for _ in range(self.negative_samples):
                    negative_item_id = negative_items[int(rng.integers(len(negative_items)))]
                    negative_index = self._item_index[negative_item_id]
                    user_vector = self._user_factors[user_index].copy()
                    positive_vector = self._item_factors[positive_index].copy()
                    negative_vector = self._item_factors[negative_index].copy()
                    difference = (
                        self._item_bias[positive_index]
                        + float(np.dot(user_vector, positive_vector))
                        - self._item_bias[negative_index]
                        - float(np.dot(user_vector, negative_vector))
                    )
                    gradient = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, difference))))
                    rate = self.learning_rate
                    regularization = self.regularization
                    self._user_factors[user_index] += rate * (
                        gradient * (positive_vector - negative_vector)
                        - regularization * user_vector
                    )
                    self._item_factors[positive_index] += rate * (
                        gradient * user_vector - regularization * positive_vector
                    )
                    self._item_factors[negative_index] += rate * (
                        -gradient * user_vector - regularization * negative_vector
                    )
                    self._item_bias[positive_index] += rate * (
                        gradient - regularization * self._item_bias[positive_index]
                    )
                    self._item_bias[negative_index] += rate * (
                        -gradient - regularization * self._item_bias[negative_index]
                    )

        self._positive_items_by_user = {
            user_id: frozenset(item_ids) for user_id, item_ids in positive_items_by_user.items()
        }
        self._history_by_user = {
            user_id: frozenset(item_ids) for user_id, item_ids in history_by_user.items()
        }
        self._popularity = popularity
        self._catalog = frozenset(catalog)
        self._cutoff_timestamp = cutoff_timestamp
        self._snapshot_id = snapshot_id
        self._model_hash = stable_model_hash(
            {
                "model_id": self.model_id,
                "embedding_dimensions": self.embedding_dimensions,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "positive_threshold": self.positive_threshold,
                "epochs": self.epochs,
                "negative_samples": self.negative_samples,
                "seed": self.seed,
                "cutoff_timestamp": timestamp_value(cutoff_timestamp),
                "snapshot_id": snapshot_id,
                "catalog": list(catalog),
                "positive_pairs": positive_pairs,
                "user_factors": self._user_factors.tolist(),
                "item_factors": self._item_factors.tolist(),
                "item_bias": self._item_bias.tolist(),
            }
        )
        return self

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            raise RuntimeError("BPRMFRanker must be fitted before reading model_hash")
        return self._model_hash

    def rank(self, candidate_set: CandidateSet, k: int | None = None) -> RankingResult:
        """Rank the exact candidates from a shared candidate set."""

        self._require_fitted(candidate_set)
        _validate_k(k)
        self._validate_candidate_set(candidate_set)
        if not candidate_set.candidate_item_ids:
            return self._result(candidate_set, (), "no_candidates")

        positive_history = self._positive_history(candidate_set)
        if not positive_history:
            return self._popularity_result(candidate_set, k, "no_positive_history")

        scored = [
            (
                item_id,
                float(
                    self._item_bias[self._item_index[item_id]]
                    + np.dot(
                        self._user_factors[self._user_index[candidate_set.user_id]],
                        self._item_factors[self._item_index[item_id]],
                    )
                ),
            )
            for item_id in candidate_set.candidate_item_ids
        ]
        scored.sort(key=lambda row: (-row[1], row[0]))
        if k is not None:
            scored = scored[:k]
        items = tuple(
            RankedItem(
                item_id=item_id,
                rank=rank,
                total_score=score,
                component_scores={"mf": score},
            )
            for rank, (item_id, score) in enumerate(scored, start=1)
        )
        return self._result(candidate_set, items, None)

    def _positive_history(self, candidate_set: CandidateSet) -> frozenset[str]:
        fitted_history = self._positive_items_by_user.get(candidate_set.user_id, frozenset())
        if candidate_set.history_item_ids:
            return fitted_history & candidate_set.history_item_ids
        return fitted_history

    def _popularity_result(
        self,
        candidate_set: CandidateSet,
        k: int | None,
        fallback_reason: str,
    ) -> RankingResult:
        ordered = sorted(
            candidate_set.candidate_item_ids,
            key=lambda item_id: (-self._popularity.get(item_id, 0), item_id),
        )
        if k is not None:
            ordered = ordered[:k]
        items = tuple(
            RankedItem(
                item_id=item_id,
                rank=rank,
                total_score=float(self._popularity.get(item_id, 0)),
                component_scores={
                    "mf": 0.0,
                    "popularity": float(self._popularity.get(item_id, 0)),
                },
            )
            for rank, item_id in enumerate(ordered, start=1)
        )
        return self._result(candidate_set, items, fallback_reason)

    def _result(
        self,
        candidate_set: CandidateSet,
        items: tuple[RankedItem, ...],
        fallback_reason: str | None,
    ) -> RankingResult:
        return RankingResult(
            model_id=self.model_id,
            user_id=candidate_set.user_id,
            snapshot_id=candidate_set.snapshot_id,
            cutoff_timestamp=candidate_set.cutoff_timestamp,
            candidate_item_ids=candidate_set.candidate_item_ids,
            items=items,
            configuration={
                "embedding_dimensions": self.embedding_dimensions,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "positive_threshold": self.positive_threshold,
                "epochs": self.epochs,
                "negative_samples": self.negative_samples,
                "cutoff_rule": "timestamp_strictly_before",
                "negative_rule": "unknown_items_only",
                "tie_break": "item_id_ascending",
            },
            seed=self.seed,
            model_hash=self._model_hash or "",
            fallback_reason=fallback_reason,
        )

    def _validate_candidate_set(self, candidate_set: CandidateSet) -> None:
        if self._snapshot_id is not None and candidate_set.snapshot_id != self._snapshot_id:
            raise ValueError("candidate snapshot does not match fitted snapshot")
        user_history = self._history_by_user.get(candidate_set.user_id, frozenset())
        if set(candidate_set.candidate_item_ids) & user_history:
            raise ValueError("candidate_item_ids must exclude history items")
        if set(candidate_set.candidate_item_ids) - self._catalog:
            raise ValueError("candidate_item_ids must be known before the cutoff")

    def _require_fitted(self, candidate_set: CandidateSet) -> None:
        if self._cutoff_timestamp is None or self._model_hash is None:
            raise RuntimeError("BPRMFRanker must be fitted before ranking")
        if timestamp_value(candidate_set.cutoff_timestamp) != timestamp_value(
            self._cutoff_timestamp
        ):
            raise ValueError("candidate cutoff does not match fitted cutoff")


def rank_bpr_mf(
    interactions: Iterable[Mapping[str, Any]] | Any,
    candidate_set: CandidateSet,
    *,
    embedding_dimensions: int = 64,
    learning_rate: float = 0.001,
    regularization: float = 0.0001,
    positive_threshold: float = 4.0,
    epochs: int = 50,
    negative_samples: int = 1,
    snapshot_id: str | None = None,
    seed: int | None = 7,
    k: int | None = None,
) -> RankingResult:
    """Fit and apply BPR matrix factorization in one call."""

    return (
        BPRMFRanker(
            embedding_dimensions=embedding_dimensions,
            learning_rate=learning_rate,
            regularization=regularization,
            positive_threshold=positive_threshold,
            epochs=epochs,
            negative_samples=negative_samples,
            seed=seed,
        )
        .fit(interactions, candidate_set.cutoff_timestamp, snapshot_id=snapshot_id)
        .rank(candidate_set, k=k)
    )


BPRMF = BPRMFRanker
rank_bpr = rank_bpr_mf
