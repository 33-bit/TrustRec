"""Full-catalog H0 and T0 scoring for many users from fitted components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata

from trustrec.explanations.evidence import aggregate_evidence, compute_aspect_priors
from trustrec.nlp.records import ASPECTS

from .base import MatrixRanker, TopKRecommendation
from .contracts import Timestamp, timestamp_value
from .matrix import InteractionMatrix, top_k_order

HYBRID_MODEL_IDS = ("h0_fixed_hybrid", "t0_trustrec")


def percentile_vector(values: np.ndarray) -> np.ndarray:
    """Vector form of ``hybrid.percentile_normalize`` with average-rank ties."""

    if values.size == 0:
        return values.astype(np.float64)
    if np.all(values == values[0]):
        return np.full(values.size, 0.5)
    ranks = rankdata(values, method="average") - 1.0
    return ranks / max(values.size - 1, 1)


@dataclass
class AspectModel:
    """Item aspect profiles and user aspect weights before one cutoff."""

    aspects: tuple[str, ...]
    item_deviation: np.ndarray
    item_support: np.ndarray
    user_counts: dict[int, np.ndarray]
    prior_distribution: np.ndarray
    alpha: float
    evidence_rows: int
    profiled_items: int

    def user_weights(self, user_index: int | None) -> np.ndarray:
        counts = self.user_counts.get(user_index) if user_index is not None else None
        if counts is None:
            counts = np.zeros(len(self.aspects))
        return (counts + self.alpha * self.prior_distribution) / (counts.sum() + self.alpha)

    def scores(self, user_index: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``A(u, i)`` and ``C(u, i)`` for every catalog item."""

        weights = self.user_weights(user_index)
        return self.item_deviation @ weights, self.item_support @ weights


def build_aspect_model(
    evidence_rows: Iterable[Mapping[str, Any]],
    matrix: InteractionMatrix,
    cutoff_timestamp: Timestamp,
    *,
    shrinkage_lambda: float = 5.0,
    alpha: float = 1.0,
    aspects: Sequence[str] = ASPECTS,
) -> AspectModel:
    """Aggregate pre-cutoff evidence into dense item and user aspect arrays.

    Item scores shrink toward the aspect prior ``mu_a``; items without evidence
    keep ``v_ia = mu_a`` and ``C_ia = 0``. User weights follow
    ``w_ua = (n_ua + alpha * pi_a) / (sum_b n_ub + alpha)`` where ``n_ua``
    counts the user's reviews that mention aspect ``a`` and ``pi_a`` is the
    share of review-aspect mentions in the evidence.
    """

    cutoff = timestamp_value(cutoff_timestamp)
    rows = [row for row in evidence_rows if row.get("aspect") in aspects]
    if any(timestamp_value(row["timestamp"]) >= cutoff for row in rows):
        raise ValueError("evidence rows must be strictly before the cutoff")
    aspect_index = {aspect: position for position, aspect in enumerate(aspects)}
    priors = compute_aspect_priors(rows, cutoff_timestamp=cutoff_timestamp, aspects=aspects)
    mu = np.array([priors.get(aspect, 0.0) for aspect in aspects])
    profiles = aggregate_evidence(
        rows,
        cutoff_timestamp=cutoff_timestamp,
        prior_scores=priors,
        shrinkage_lambda=shrinkage_lambda,
    )
    deviation = np.zeros((matrix.n_items, len(aspects)))
    support = np.zeros((matrix.n_items, len(aspects)))
    profiled: set[int] = set()
    for (item_id, aspect), profile in profiles.items():
        item = matrix.item_index.get(item_id)
        if item is None:
            continue
        column = aspect_index[aspect]
        deviation[item, column] = profile.score - mu[column]
        support[item, column] = profile.support
        profiled.add(item)

    mentions: set[tuple[str, str, str]] = set()
    for row in rows:
        author = row.get("author_id", row.get("user_id"))
        if author is not None:
            mentions.add((str(author), str(row["review_id"]), row["aspect"]))
    user_counts: dict[int, np.ndarray] = {}
    totals = np.zeros(len(aspects))
    for author, _, aspect in mentions:
        totals[aspect_index[aspect]] += 1.0
        user = matrix.user_index.get(author)
        if user is None:
            continue
        counts = user_counts.setdefault(user, np.zeros(len(aspects)))
        counts[aspect_index[aspect]] += 1.0
    prior_distribution = (
        totals / totals.sum() if totals.sum() > 0 else np.full(len(aspects), 1.0 / len(aspects))
    )
    return AspectModel(
        aspects=tuple(aspects),
        item_deviation=deviation,
        item_support=support,
        user_counts=user_counts,
        prior_distribution=prior_distribution,
        alpha=float(alpha),
        evidence_rows=len(rows),
        profiled_items=len(profiled),
    )


@dataclass(frozen=True)
class HybridConfig:
    """One H0 or T0 setting."""

    model_id: str
    mf_weight: float = 0.5
    graph_weight: float = 0.25
    aspect_weight: float = 0.25
    kappa: float = 5.0
    g_max: float = 0.5
    rho: float = 0.5

    def __post_init__(self) -> None:
        if self.model_id not in HYBRID_MODEL_IDS:
            raise ValueError(f"unsupported hybrid model_id: {self.model_id}")
        if self.model_id == "h0_fixed_hybrid":
            weights = (self.mf_weight, self.graph_weight, self.aspect_weight)
            if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
                raise ValueError("H0 weights must be non-negative and sum to 1")
        else:
            if self.kappa <= 0 or not 0.0 <= self.g_max <= 1.0 or not 0.0 <= self.rho <= 1.0:
                raise ValueError("invalid T0 gate parameters")

    def parameters(self) -> dict[str, float]:
        if self.model_id == "h0_fixed_hybrid":
            return {
                "mf_weight": self.mf_weight,
                "graph_weight": self.graph_weight,
                "aspect_weight": self.aspect_weight,
            }
        return {"kappa": self.kappa, "g_max": self.g_max, "rho": self.rho}


@dataclass
class UserComponents:
    candidates: np.ndarray
    mf: np.ndarray
    graph: np.ndarray
    aspect: np.ndarray
    support: np.ndarray
    history_count: int
    mf_fallback: str | None
    graph_fallback: str | None


class HybridBatchScorer:
    """Score H0 and T0 for many users from shared fitted components.

    Each component covers the protocol candidate pool (catalog minus history).
    MF and graph components use the same popularity fallback as B2 and B3, so
    H0 and T0 differ from the baselines only through the hybrid formula.
    """

    def __init__(
        self,
        matrix: InteractionMatrix,
        mf_ranker: MatrixRanker,
        graph_ranker: MatrixRanker,
        aspect_model: AspectModel,
    ) -> None:
        for ranker in (mf_ranker, graph_ranker):
            if ranker.matrix.fingerprint != matrix.fingerprint:
                raise ValueError("component rankers must use the hybrid matrix")
        self.matrix = matrix
        self.mf_ranker = mf_ranker
        self.graph_ranker = graph_ranker
        self.aspect_model = aspect_model

    def prepare(self, user_ids: Sequence[str]) -> None:
        prepare = getattr(self.graph_ranker, "prepare", None)
        if prepare is not None:
            prepare(user_ids)

    def clear_cache(self) -> None:
        clear = getattr(self.graph_ranker, "clear_cache", None)
        if clear is not None:
            clear()

    def _component(
        self, ranker: MatrixRanker, user_id: str, positives: np.ndarray, candidates: np.ndarray
    ) -> tuple[np.ndarray, str | None]:
        popularity = self.matrix.popularity[candidates].astype(np.float64)
        if positives.size == 0:
            return popularity, "no_positive_history"
        scores = ranker._user_scores(user_id, positives)[candidates]
        if ranker.zero_score_fallback is not None and not (scores > 0.0).any():
            return popularity, ranker.zero_score_fallback
        return scores, None

    def components(self, user_id: str) -> UserComponents:
        matrix = self.matrix
        candidates = matrix.candidate_indices(user_id)
        positives = matrix.positive_indices(user_id)
        mf, mf_fallback = self._component(self.mf_ranker, user_id, positives, candidates)
        graph, graph_fallback = self._component(self.graph_ranker, user_id, positives, candidates)
        aspect, support = self.aspect_model.scores(matrix.user_index.get(user_id))
        return UserComponents(
            candidates=candidates,
            mf=percentile_vector(mf),
            graph=percentile_vector(graph),
            aspect=percentile_vector(aspect[candidates]),
            support=support[candidates],
            history_count=int(matrix.history_indices(user_id).size),
            mf_fallback=mf_fallback,
            graph_fallback=graph_fallback,
        )

    @staticmethod
    def totals(parts: UserComponents, config: HybridConfig) -> tuple[np.ndarray, np.ndarray]:
        """Return total scores and per-candidate gates for one configuration."""

        if config.model_id == "h0_fixed_hybrid":
            total = (
                config.mf_weight * parts.mf
                + config.graph_weight * parts.graph
                + config.aspect_weight * parts.aspect
            )
            return total, np.zeros(parts.candidates.size)
        history_factor = config.kappa / (config.kappa + parts.history_count)
        gate = config.g_max * history_factor * parts.support
        base = config.rho * parts.mf + (1.0 - config.rho) * parts.graph
        return (1.0 - gate) * base + gate * parts.aspect, gate

    def recommend_many_configs(
        self, user_id: str, configs: Sequence[HybridConfig], k: int
    ) -> list[tuple[TopKRecommendation, float]]:
        """Return one top-K result and mean gate per configuration for one user."""

        parts = self.components(user_id)
        if parts.candidates.size == 0:
            empty = TopKRecommendation(user_id, (), (), 0, "", "no_candidates")
            return [(empty, 0.0) for _ in configs]
        signature = self.matrix.candidate_signature(user_id)
        fallback = parts.mf_fallback if parts.mf_fallback == parts.graph_fallback else None
        results = []
        for config in configs:
            total, gate = self.totals(parts, config)
            order = top_k_order(total, parts.candidates, k)
            results.append(
                (
                    TopKRecommendation(
                        user_id=user_id,
                        item_ids=tuple(
                            self.matrix.item_ids[int(parts.candidates[p])] for p in order
                        ),
                        scores=tuple(float(total[p]) for p in order),
                        candidate_count=int(parts.candidates.size),
                        candidate_signature=signature,
                        fallback_reason=fallback,
                    ),
                    float(gate.mean()) if gate.size else 0.0,
                )
            )
        return results


# A forked worker reads the scorer from this module global.
_WORKER_STATE: dict[str, Any] = {}


def _score_chunk(
    scorer: HybridBatchScorer,
    user_ids: Sequence[str],
    configs: Sequence[HybridConfig],
    k: int,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    import time

    rows: list[list[dict[str, Any]]] = [[] for _ in configs]
    for start in range(0, len(user_ids), batch_size):
        chunk = list(user_ids[start : start + batch_size])
        began = time.perf_counter()
        scorer.prepare(chunk)
        shared_ms = (time.perf_counter() - began) * 1000.0 / max(len(chunk), 1)
        for user_id in chunk:
            began = time.perf_counter()
            results = scorer.recommend_many_configs(user_id, configs, k)
            elapsed_ms = (time.perf_counter() - began) * 1000.0 + shared_ms
            for position, (recommendation, mean_gate) in enumerate(results):
                rows[position].append(
                    {
                        "user_id": user_id,
                        "ranked_item_ids": list(recommendation.item_ids),
                        "scores": list(recommendation.scores),
                        "candidate_rule": "catalog_minus_history",
                        "candidate_count": recommendation.candidate_count,
                        "candidate_signature": recommendation.candidate_signature,
                        "fallback_reason": recommendation.fallback_reason,
                        "mean_gate": mean_gate,
                        "latency_ms": elapsed_ms,
                    }
                )
        scorer.clear_cache()
    return rows


def _worker(user_ids: list[str]) -> list[list[dict[str, Any]]]:
    state = _WORKER_STATE
    return _score_chunk(state["scorer"], user_ids, state["configs"], state["k"], state["batch"])


def score_hybrid_configs(
    scorer: HybridBatchScorer,
    user_ids: Sequence[str],
    configs: Sequence[HybridConfig],
    k: int,
    *,
    workers: int = 1,
    batch_size: int = 64,
) -> list[list[dict[str, Any]]]:
    """Return one list of top-K rows per configuration, users in input order.

    Components are computed once per user and shared by every configuration.
    """

    import multiprocessing

    if workers <= 1 or len(user_ids) < 2:
        return _score_chunk(scorer, user_ids, configs, k, batch_size)
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("workers > 1 requires the fork start method")
    chunk_size = max(1, -(-len(user_ids) // (workers * 8)))
    chunks = [list(user_ids[i : i + chunk_size]) for i in range(0, len(user_ids), chunk_size)]
    _WORKER_STATE.update(scorer=scorer, configs=list(configs), k=k, batch=batch_size)
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_worker, chunks, chunksize=1)
    finally:
        _WORKER_STATE.clear()
    return [[row for part in parts for row in part[position]] for position in range(len(configs))]
