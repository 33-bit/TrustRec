"""Bayesian personalized ranking matrix-factorization baseline."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .base import MatrixRanker
from .contracts import CandidateSet, RankingResult
from .matrix import InteractionMatrix

_MAX_REJECTION_ROUNDS = 20


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class BPRMFRanker(MatrixRanker):
    """Rank candidates with seeded BPR matrix factorization.

    Training uses mini-batch stochastic gradient ascent on the BPR objective.
    Each batch samples one unknown item per positive pair and negative sample
    and applies summed updates, so a batch size of one equals classic SGD.
    """

    model_id = "b2_bpr_mf"
    score_component = "mf"

    def __init__(
        self,
        embedding_dimensions: int = 64,
        learning_rate: float = 0.001,
        regularization: float = 0.0001,
        positive_threshold: float = 4.0,
        epochs: int = 50,
        negative_samples: int = 1,
        seed: int | None = 7,
        batch_size: int = 1024,
    ) -> None:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("seed must be an integer or None")
        super().__init__(
            positive_threshold=positive_threshold, seed=int(seed) if seed is not None else None
        )
        self.embedding_dimensions = _positive_integer("embedding_dimensions", embedding_dimensions)
        self.epochs = _positive_integer("epochs", epochs)
        self.negative_samples = _positive_integer("negative_samples", negative_samples)
        self.batch_size = _positive_integer("batch_size", batch_size)
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(regularization)) or regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self._factor_rows = np.empty(0, dtype=np.int64)
        self._user_factors = np.empty((0, self.embedding_dimensions), dtype=np.float64)
        self._item_factors = np.empty((0, self.embedding_dimensions), dtype=np.float64)
        self._item_bias = np.empty(0, dtype=np.float64)

    def _configuration(self) -> dict[str, Any]:
        return {
            "embedding_dimensions": self.embedding_dimensions,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "positive_threshold": self.positive_threshold,
            "epochs": self.epochs,
            "negative_samples": self.negative_samples,
            "batch_size": self.batch_size,
            "optimizer": "minibatch_sgd_summed_updates",
            "cutoff_rule": "timestamp_strictly_before",
            "negative_rule": "unknown_items_only",
            "tie_break": "item_id_ascending",
        }

    def _hash_configuration(self) -> dict[str, Any]:
        configuration = self._configuration()
        return {
            key: configuration[key]
            for key in (
                "embedding_dimensions",
                "learning_rate",
                "regularization",
                "positive_threshold",
                "epochs",
                "negative_samples",
                "batch_size",
                "optimizer",
            )
        }

    def init_parameters(self) -> dict[str, Any]:
        return {
            "embedding_dimensions": self.embedding_dimensions,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "positive_threshold": self.positive_threshold,
            "epochs": self.epochs,
            "negative_samples": self.negative_samples,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }

    def _state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "factor_rows": self._factor_rows,
            "user_factors": self._user_factors,
            "item_factors": self._item_factors,
            "item_bias": self._item_bias,
        }

    def _restore_state(self, matrix: InteractionMatrix, arrays: Mapping[str, np.ndarray]) -> None:
        factor_rows = np.asarray(arrays["factor_rows"], dtype=np.int64)
        user_factors = np.asarray(arrays["user_factors"], dtype=np.float64)
        item_factors = np.asarray(arrays["item_factors"], dtype=np.float64)
        item_bias = np.asarray(arrays["item_bias"], dtype=np.float64)
        dimensions = self.embedding_dimensions
        if (
            factor_rows.shape != (matrix.n_users,)
            or item_factors.shape != (matrix.n_items, dimensions)
            or item_bias.shape != (matrix.n_items,)
            or user_factors.ndim != 2
            or user_factors.shape[1] != dimensions
            or factor_rows.max(initial=-1) >= user_factors.shape[0]
        ):
            raise ValueError("saved BPR arrays do not match the matrix or configuration")
        self._factor_rows = factor_rows
        self._user_factors = user_factors
        self._item_factors = item_factors
        self._item_bias = item_bias

    def _learned_state_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (self._factor_rows, self._user_factors, self._item_factors, self._item_bias):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def _fit_matrix(self, matrix: InteractionMatrix) -> None:
        n_items = matrix.n_items
        positive = matrix.positive.tocoo()
        pair_users = positive.row.astype(np.int64)
        pair_items = positive.col.astype(np.int64)
        order = np.lexsort((pair_items, pair_users))
        pair_users = pair_users[order]
        pair_items = pair_items[order]

        users_with_positives = np.unique(pair_users)
        self._factor_rows = np.full(matrix.n_users, -1, dtype=np.int64)
        self._factor_rows[users_with_positives] = np.arange(users_with_positives.size)
        rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        self._user_factors = rng.normal(
            0.0, 0.01, size=(users_with_positives.size, self.embedding_dimensions)
        )
        self._item_factors = rng.normal(0.0, 0.01, size=(n_items, self.embedding_dimensions))
        self._item_bias = np.zeros(n_items, dtype=np.float64)

        history = matrix.history
        history_counts = np.diff(history.indptr)
        trainable = history_counts[pair_users] < n_items
        pair_users = pair_users[trainable]
        pair_items = pair_items[trainable]
        if pair_users.size == 0:
            return
        history_rows = np.repeat(np.arange(matrix.n_users, dtype=np.int64), history_counts)
        history_keys = history_rows * n_items + history.indices.astype(np.int64)

        for _ in range(self.epochs):
            permutation = rng.permutation(pair_users.size)
            for start in range(0, permutation.size, self.batch_size):
                batch = permutation[start : start + self.batch_size]
                users = pair_users[batch]
                positives = pair_items[batch]
                for _ in range(self.negative_samples):
                    negatives = _sample_negatives(rng, users, history, history_keys, n_items)
                    self._update(users, positives, negatives)

    def _update(self, users: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> None:
        rows = self._factor_rows[users]
        user_vectors = self._user_factors[rows]
        positive_vectors = self._item_factors[positives]
        negative_vectors = self._item_factors[negatives]
        difference = (
            self._item_bias[positives]
            - self._item_bias[negatives]
            + np.einsum("ij,ij->i", user_vectors, positive_vectors - negative_vectors)
        )
        gradient = 1.0 / (1.0 + np.exp(np.clip(difference, -50.0, 50.0)))
        rate = self.learning_rate
        regularization = self.regularization
        weight = gradient[:, None]
        np.add.at(
            self._user_factors,
            rows,
            rate * (weight * (positive_vectors - negative_vectors) - regularization * user_vectors),
        )
        np.add.at(
            self._item_factors,
            positives,
            rate * (weight * user_vectors - regularization * positive_vectors),
        )
        np.add.at(
            self._item_factors,
            negatives,
            rate * (-weight * user_vectors - regularization * negative_vectors),
        )
        np.add.at(
            self._item_bias,
            positives,
            rate * (gradient - regularization * self._item_bias[positives]),
        )
        np.add.at(
            self._item_bias,
            negatives,
            rate * (-gradient - regularization * self._item_bias[negatives]),
        )

    def _user_scores(self, user_id: str, positive_items: np.ndarray) -> np.ndarray:
        row = self._factor_rows[self.matrix.user_index[user_id]]
        return self._item_bias + self._item_factors @ self._user_factors[row]


def _sample_negatives(
    rng: np.random.Generator,
    users: np.ndarray,
    history: Any,
    history_keys: np.ndarray,
    n_items: int,
) -> np.ndarray:
    """Sample one item per user uniformly from items outside that user's history."""

    negatives = rng.integers(0, n_items, size=users.size)
    pending = np.arange(users.size)
    for _ in range(_MAX_REJECTION_ROUNDS):
        keys = users[pending] * n_items + negatives[pending]
        found = np.minimum(np.searchsorted(history_keys, keys), history_keys.size - 1)
        pending = pending[history_keys[found] == keys]
        if pending.size == 0:
            return negatives
        negatives[pending] = rng.integers(0, n_items, size=pending.size)
    for position in pending:
        user = users[position]
        seen = history.indices[history.indptr[user] : history.indptr[user + 1]]
        unknown = np.setdiff1d(np.arange(n_items), seen, assume_unique=True)
        negatives[position] = unknown[int(rng.integers(unknown.size))]
    return negatives


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
    batch_size: int = 1024,
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
            batch_size=batch_size,
        )
        .fit(interactions, candidate_set.cutoff_timestamp, snapshot_id=snapshot_id)
        .rank(candidate_set, k=k)
    )
