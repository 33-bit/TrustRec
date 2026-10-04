"""Most-popular ranking baseline."""

from __future__ import annotations

import math
from collections import Counter
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


def _rating_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("rating")
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("rating must be numeric when provided")
    try:
        rating = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("rating must be numeric when provided") from error
    if not math.isfinite(rating):
        raise ValueError("rating must be finite")
    return rating


class PopularityRanker:
    """Rank candidates by positive interactions before one cutoff."""

    model_id = "b0_most_popular"

    def __init__(self, positive_threshold: float = 4.0, seed: int | None = None) -> None:
        if not math.isfinite(float(positive_threshold)):
            raise ValueError("positive_threshold must be finite")
        self.positive_threshold = float(positive_threshold)
        self.seed = seed
        self._counts: Counter[str] = Counter()
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
    ) -> PopularityRanker:
        """Fit positive counts using only rows strictly before the cutoff."""

        rows = first_interactions_before_cutoff(interactions, cutoff_timestamp)
        counts: Counter[str] = Counter()
        history_by_user: dict[str, set[str]] = {}
        for row in rows:
            history_by_user.setdefault(row["user_id"], set()).add(row["item_id"])
            rating = _rating_value(row)
            if rating is not None and rating >= self.positive_threshold:
                counts[row["item_id"]] += 1
        self._counts = counts
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
                "cutoff_timestamp": timestamp_value(cutoff_timestamp),
                "snapshot_id": snapshot_id,
                "catalog": sorted(self._catalog),
                "counts": dict(sorted(counts.items())),
            }
        )
        return self

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            raise RuntimeError("PopularityRanker must be fitted before reading model_hash")
        return self._model_hash

    def rank(self, candidate_set: CandidateSet, k: int | None = None) -> RankingResult:
        """Rank the exact candidates from a shared candidate set."""

        self._require_fitted(candidate_set)
        _validate_k(k)
        ordered = sorted(
            candidate_set.candidate_item_ids,
            key=lambda item_id: (-self._counts.get(item_id, 0), item_id),
        )
        if k is not None:
            ordered = ordered[:k]
        items = tuple(
            RankedItem(
                item_id=item_id,
                rank=rank,
                total_score=float(self._counts.get(item_id, 0)),
                component_scores={"popularity": float(self._counts.get(item_id, 0))},
            )
            for rank, item_id in enumerate(ordered, start=1)
        )
        return RankingResult(
            model_id=self.model_id,
            user_id=candidate_set.user_id,
            snapshot_id=candidate_set.snapshot_id,
            cutoff_timestamp=candidate_set.cutoff_timestamp,
            candidate_item_ids=candidate_set.candidate_item_ids,
            items=items,
            configuration={
                "positive_threshold": self.positive_threshold,
                "cutoff_rule": "timestamp_strictly_before",
                "tie_break": "item_id_ascending",
            },
            seed=self.seed,
            model_hash=self._model_hash or "",
            fallback_reason="no_candidates" if not candidate_set.candidate_item_ids else None,
        )

    def _require_fitted(self, candidate_set: CandidateSet) -> None:
        if self._cutoff_timestamp is None or self._model_hash is None:
            raise RuntimeError("PopularityRanker must be fitted before ranking")
        if timestamp_value(candidate_set.cutoff_timestamp) != timestamp_value(
            self._cutoff_timestamp
        ):
            raise ValueError("candidate cutoff does not match fitted cutoff")
        if self._snapshot_id is not None and candidate_set.snapshot_id != self._snapshot_id:
            raise ValueError("candidate snapshot does not match fitted snapshot")
        user_history = self._history_by_user.get(candidate_set.user_id, frozenset())
        if set(candidate_set.candidate_item_ids) & user_history:
            raise ValueError("candidate_item_ids must exclude history items")
        if set(candidate_set.candidate_item_ids) - self._catalog:
            raise ValueError("candidate_item_ids must be known before the cutoff")


def _validate_k(k: int | None) -> None:
    if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
        raise ValueError("k must be a positive integer or None")


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


MostPopularRanker = PopularityRanker
rank_most_popular = rank_popularity
