"""Personalized PageRank on a positive user-item interaction graph."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import numpy as np

from trustrec.recommenders.contracts import (
    CandidateSet,
    RankedItem,
    RankingResult,
    Timestamp,
    first_interactions_before_cutoff,
    stable_model_hash,
    timestamp_value,
)
from trustrec.recommenders.popularity import _rating_value, _validate_k


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class PersonalizedPageRankRanker:
    """Rank candidates by PageRank mass reached from one user node."""

    model_id = "b3_ppr"

    def __init__(
        self,
        damping: float = 0.85,
        positive_threshold: float = 4.0,
        max_iter: int = 100,
        tol: float = 1e-12,
        seed: int | None = None,
    ) -> None:
        if not math.isfinite(float(damping)) or not 0 <= damping < 1:
            raise ValueError("damping must be finite and in [0, 1)")
        if not math.isfinite(float(positive_threshold)):
            raise ValueError("positive_threshold must be finite")
        if not math.isfinite(float(tol)) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        self.damping = float(damping)
        self.positive_threshold = float(positive_threshold)
        self.max_iter = _positive_integer("max_iter", max_iter)
        self.tol = float(tol)
        self.seed = seed
        self._user_to_items: dict[str, frozenset[str]] = {}
        self._item_to_users: dict[str, frozenset[str]] = {}
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
    ) -> PersonalizedPageRankRanker:
        """Build positive graph edges from first pre-cutoff events."""

        rows = first_interactions_before_cutoff(interactions, cutoff_timestamp)
        user_to_items: defaultdict[str, set[str]] = defaultdict(set)
        item_to_users: defaultdict[str, set[str]] = defaultdict(set)
        history_by_user: dict[str, set[str]] = {}
        popularity: Counter[str] = Counter()
        for row in rows:
            user_id = row["user_id"]
            item_id = row["item_id"]
            history_by_user.setdefault(user_id, set()).add(item_id)
            rating = _rating_value(row)
            if rating is None or rating < self.positive_threshold:
                continue
            if item_id in user_to_items[user_id]:
                continue
            user_to_items[user_id].add(item_id)
            item_to_users[item_id].add(user_id)
            popularity[item_id] += 1
        self._user_to_items = {
            user_id: frozenset(item_ids) for user_id, item_ids in user_to_items.items()
        }
        self._item_to_users = {
            item_id: frozenset(user_ids) for item_id, user_ids in item_to_users.items()
        }
        self._history_by_user = {
            user_id: frozenset(item_ids) for user_id, item_ids in history_by_user.items()
        }
        self._popularity = popularity
        self._catalog = frozenset(row["item_id"] for row in rows)
        self._cutoff_timestamp = cutoff_timestamp
        self._snapshot_id = snapshot_id
        self._model_hash = stable_model_hash(
            {
                "model_id": self.model_id,
                "damping": self.damping,
                "positive_threshold": self.positive_threshold,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "seed": self.seed,
                "cutoff_timestamp": timestamp_value(cutoff_timestamp),
                "snapshot_id": snapshot_id,
                "catalog": sorted(self._catalog),
                "user_to_items": {
                    user_id: sorted(item_ids)
                    for user_id, item_ids in sorted(self._user_to_items.items())
                },
            }
        )
        return self

    @property
    def model_hash(self) -> str:
        if self._model_hash is None:
            raise RuntimeError(
                "PersonalizedPageRankRanker must be fitted before reading model_hash"
            )
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

        scores = self._scores_for_user(candidate_set.user_id)
        scored = [
            (item_id, float(scores.get(item_id, 0.0)))
            for item_id in candidate_set.candidate_item_ids
        ]
        if not any(score > 0.0 for _, score in scored):
            return self._popularity_result(candidate_set, k, "no_graph_path")
        scored.sort(key=lambda row: (-row[1], row[0]))
        if k is not None:
            scored = scored[:k]
        items = tuple(
            RankedItem(
                item_id=item_id,
                rank=rank,
                total_score=score,
                component_scores={"ppr": score},
            )
            for rank, (item_id, score) in enumerate(scored, start=1)
        )
        return self._result(candidate_set, items, None)

    def _scores_for_user(self, user_id: str) -> dict[str, float]:
        positive_items = self._user_to_items.get(user_id, frozenset())
        if not positive_items:
            return {}
        nodes = tuple(
            [("user", user_id) for user_id in sorted(self._user_to_items)]
            + [("item", item_id) for item_id in sorted(self._item_to_users)]
        )
        node_index = {node: index for index, node in enumerate(nodes)}
        adjacency: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
        for node_type, node_id in nodes:
            if node_type == "user":
                adjacency[(node_type, node_id)] = tuple(
                    ("item", item_id) for item_id in sorted(self._user_to_items[node_id])
                )
            else:
                adjacency[(node_type, node_id)] = tuple(
                    ("user", user_id) for user_id in sorted(self._item_to_users[node_id])
                )
        restart_node = ("user", user_id)
        if restart_node not in node_index:
            return {}
        scores = np.zeros(len(nodes), dtype=np.float64)
        scores[node_index[restart_node]] = 1.0
        restart_index = node_index[restart_node]
        for _ in range(self.max_iter):
            updated = np.zeros(len(nodes), dtype=np.float64)
            updated[restart_index] = 1.0 - self.damping
            dangling_mass = 0.0
            for node, node_position in node_index.items():
                probability = scores[node_position]
                neighbors = adjacency[node]
                if not neighbors:
                    dangling_mass += probability
                    continue
                share = self.damping * probability / len(neighbors)
                for neighbor in neighbors:
                    updated[node_index[neighbor]] += share
            updated[restart_index] += self.damping * dangling_mass
            if float(np.max(np.abs(updated - scores))) <= self.tol:
                scores = updated
                break
            scores = updated
        return {
            item_id: float(scores[node_index[("item", item_id)]])
            for item_id in self._item_to_users
            if ("item", item_id) in node_index
        }

    def _positive_history(self, candidate_set: CandidateSet) -> frozenset[str]:
        fitted_history = self._user_to_items.get(candidate_set.user_id, frozenset())
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
                    "ppr": 0.0,
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
                "damping": self.damping,
                "positive_threshold": self.positive_threshold,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "cutoff_rule": "timestamp_strictly_before",
                "edge_rule": "positive_ratings_only",
                "dangling_rule": "restart_mass_to_user",
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
            raise RuntimeError("PersonalizedPageRankRanker must be fitted before ranking")
        if timestamp_value(candidate_set.cutoff_timestamp) != timestamp_value(
            self._cutoff_timestamp
        ):
            raise ValueError("candidate cutoff does not match fitted cutoff")


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


PPRRanker = PersonalizedPageRankRanker
PersonalizedPageRank = PersonalizedPageRankRanker
rank_ppr = rank_personalized_pagerank
