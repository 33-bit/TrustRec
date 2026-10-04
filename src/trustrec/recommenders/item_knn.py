"""Item-based collaborative filtering baseline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

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


def cosine_similarity(left_users: set[str], right_users: set[str]) -> float:
    """Return cosine similarity for two binary item user sets."""

    if not left_users or not right_users:
        return 0.0
    intersection = len(left_users & right_users)
    if intersection == 0:
        return 0.0
    return intersection / math.sqrt(len(left_users) * len(right_users))


class ItemKNNRanker:
    """Rank candidates by similarity to the user's positive item history."""

    model_id = "b1_item_knn"

    def __init__(
        self,
        positive_threshold: float = 4.0,
        n_neighbors: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not math.isfinite(float(positive_threshold)):
            raise ValueError("positive_threshold must be finite")
        if n_neighbors == 0:
            n_neighbors = None
        if n_neighbors is not None and (
            not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be positive, zero, or None")
        self.positive_threshold = float(positive_threshold)
        self.n_neighbors = n_neighbors
        self.seed = seed
        self._item_users: dict[str, set[str]] = {}
        self._user_items: dict[str, set[str]] = {}
        self._popularity: Counter[str] = Counter()
        self._catalog: frozenset[str] = frozenset()
        self._history_by_user: dict[str, frozenset[str]] = {}
        self._cutoff_timestamp: Timestamp | None = None
        self._snapshot_id: str | None = None
        self._model_hash: str | None = None

    def fit(
        self,
        interactions: Iterable[Mapping[str, Any]] | Any,
        cutoff_timestamp: Timestamp,
        snapshot_id: str | None = None,
    ) -> ItemKNNRanker:
        """Fit binary positive item vectors before the cutoff."""

        rows = first_interactions_before_cutoff(interactions, cutoff_timestamp)
        item_users: defaultdict[str, set[str]] = defaultdict(set)
        user_items: defaultdict[str, set[str]] = defaultdict(set)
        popularity: Counter[str] = Counter()
        history_by_user: dict[str, set[str]] = {}
        for row in rows:
            history_by_user.setdefault(row["user_id"], set()).add(row["item_id"])
            rating = _rating_value(row)
            if rating is None or rating < self.positive_threshold:
                continue
            item_id = row["item_id"]
            user_id = row["user_id"]
            if user_id in item_users[item_id]:
                continue
            item_users[item_id].add(user_id)
            user_items[user_id].add(item_id)
            popularity[item_id] += 1
        self._item_users = {item: set(users) for item, users in item_users.items()}
        self._user_items = {user: set(items) for user, items in user_items.items()}
        self._popularity = popularity
        self._catalog = frozenset(row["item_id"] for row in rows)
        self._history_by_user = {
            user_id: frozenset(item_ids) for user_id, item_ids in history_by_user.items()
        }
        self._cutoff_timestamp = cutoff_timestamp
        self._snapshot_id = snapshot_id
        self._model_hash = stable_model_hash(
            {
                "model_id": self.model_id,
                "positive_threshold": self.positive_threshold,
                "n_neighbors": self.n_neighbors,
                "cutoff_timestamp": timestamp_value(cutoff_timestamp),
                "snapshot_id": snapshot_id,
                "catalog": sorted(self._catalog),
                "item_users": {
                    item: sorted(users) for item, users in sorted(self._item_users.items())
                },
            }
        )
        return self

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            raise RuntimeError("ItemKNNRanker must be fitted before reading model_hash")
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
                        "knn": 0.0,
                        "popularity": float(self._popularity.get(item_id, 0)),
                    },
                )
                for rank, item_id in enumerate(ordered, start=1)
            )
            return self._result(candidate_set, items, "no_positive_history")

        scored: list[tuple[str, float, int]] = []
        for item_id in candidate_set.candidate_item_ids:
            similarities = [
                cosine_similarity(self._item_users.get(item_id, set()), self._item_users[history])
                for history in positive_history
            ]
            similarities = [score for score in similarities if score > 0.0]
            similarities.sort(reverse=True)
            if self.n_neighbors is not None:
                similarities = similarities[: self.n_neighbors]
            score = math.fsum(similarities)
            scored.append((item_id, score, self._popularity.get(item_id, 0)))
        scored.sort(key=lambda row: (-row[1], -row[2], row[0]))
        fallback_reason = (
            "no_similar_items" if not any(score > 0.0 for _, score, _ in scored) else None
        )
        if fallback_reason is not None:
            scored.sort(key=lambda row: (-row[2], row[0]))
        if k is not None:
            scored = scored[:k]
        items = tuple(
            RankedItem(
                item_id=item_id,
                rank=rank,
                total_score=score,
                component_scores={"knn": score, "popularity": float(popularity)},
            )
            for rank, (item_id, score, popularity) in enumerate(scored, start=1)
        )
        if fallback_reason is not None:
            items = tuple(
                RankedItem(
                    item_id=item_id,
                    rank=rank,
                    total_score=float(popularity),
                    component_scores={"knn": 0.0, "popularity": float(popularity)},
                )
                for rank, (item_id, _, popularity) in enumerate(scored, start=1)
            )
        return self._result(candidate_set, items, fallback_reason)

    def _positive_history(self, candidate_set: CandidateSet) -> set[str]:
        fitted_history = self._user_items.get(candidate_set.user_id, set())
        if candidate_set.history_item_ids:
            return fitted_history & candidate_set.history_item_ids
        return set(fitted_history)

    def _validate_candidate_set(self, candidate_set: CandidateSet) -> None:
        if self._snapshot_id is not None and candidate_set.snapshot_id != self._snapshot_id:
            raise ValueError("candidate snapshot does not match fitted snapshot")
        user_history = self._history_by_user.get(candidate_set.user_id, frozenset())
        if set(candidate_set.candidate_item_ids) & user_history:
            raise ValueError("candidate_item_ids must exclude history items")
        if set(candidate_set.candidate_item_ids) - self._catalog:
            raise ValueError("candidate_item_ids must be known before the cutoff")

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
                "positive_threshold": self.positive_threshold,
                "neighbor_limit": self.n_neighbors or 0,
                "cutoff_rule": "timestamp_strictly_before",
                "tie_break": "item_id_ascending",
            },
            seed=self.seed,
            model_hash=self._model_hash or "",
            fallback_reason=fallback_reason,
        )

    def _require_fitted(self, candidate_set: CandidateSet) -> None:
        if self._cutoff_timestamp is None or self._model_hash is None:
            raise RuntimeError("ItemKNNRanker must be fitted before ranking")
        if timestamp_value(candidate_set.cutoff_timestamp) != timestamp_value(
            self._cutoff_timestamp
        ):
            raise ValueError("candidate cutoff does not match fitted cutoff")


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


ItemKNN = ItemKNNRanker
