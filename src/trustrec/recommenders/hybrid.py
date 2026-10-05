"""Shared score normalization and TrustRec hybrid formulas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _finite_scores(scores: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(scores, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for item_id, value in scores.items():
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{name} item IDs must be non-empty strings")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} scores must be numeric") from error
        if not math.isfinite(numeric):
            raise ValueError(f"{name} scores must be finite")
        result[item_id] = numeric
    return result


def percentile_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Map scores to percentile ranks in ``[0, 1]`` with average ties."""

    values = _finite_scores(scores, "scores")
    if not values:
        return {}
    if len(set(values.values())) == 1:
        return {item_id: 0.5 for item_id in values}
    ordered = sorted(values.items(), key=lambda pair: (pair[1], pair[0]))
    normalized: dict[str, float] = {}
    index = 0
    denominator = max(len(ordered) - 1, 1)
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0
        for item_id, _ in ordered[index:end]:
            normalized[item_id] = average_rank / denominator
        index = end
    return normalized


normalize_scores = percentile_normalize
percentile_rank = percentile_normalize


def _validate_components(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    mf = _finite_scores(mf_scores, "mf_scores")
    graph = _finite_scores(graph_scores, "graph_scores")
    aspect = _finite_scores(aspect_scores, "aspect_scores")
    keys = set(mf)
    if set(graph) != keys or set(aspect) != keys:
        raise ValueError("all hybrid components must use the same candidate IDs")
    return mf, graph, aspect


def _validate_weights(mf_weight: float, graph_weight: float, aspect_weight: float) -> None:
    weights = (mf_weight, graph_weight, aspect_weight)
    if any(not math.isfinite(float(weight)) or weight < 0 for weight in weights):
        raise ValueError("hybrid weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("hybrid weights must sum to 1")


def fixed_hybrid_scores(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
    *,
    mf_weight: float = 0.5,
    graph_weight: float = 0.25,
    aspect_weight: float = 0.25,
) -> dict[str, float]:
    """Combine normalized MF, graph, and aspect scores with fixed weights."""

    _validate_weights(mf_weight, graph_weight, aspect_weight)
    mf, graph, aspect = _validate_components(mf_scores, graph_scores, aspect_scores)
    mf_rank = percentile_normalize(mf)
    graph_rank = percentile_normalize(graph)
    aspect_rank = percentile_normalize(aspect)
    return {
        item_id: mf_weight * mf_rank[item_id]
        + graph_weight * graph_rank[item_id]
        + aspect_weight * aspect_rank[item_id]
        for item_id in mf
    }


combine_fixed_hybrid = fixed_hybrid_scores
score_fixed_hybrid = fixed_hybrid_scores


def _validate_gate(kappa: float, g_max: float, rho: float) -> None:
    if not math.isfinite(float(kappa)) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if not math.isfinite(float(g_max)) or not 0.0 <= g_max <= 1.0:
        raise ValueError("g_max must be finite and between 0 and 1")
    if not math.isfinite(float(rho)) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be finite and between 0 and 1")


def adaptive_trustrec_scores(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
    support: Mapping[str, float],
    history_count: int,
    *,
    kappa: float = 5.0,
    g_max: float = 0.5,
    rho: float = 0.5,
) -> dict[str, float]:
    """Apply the adaptive TrustRec gate after shared percentile normalization."""

    _validate_gate(kappa, g_max, rho)
    if isinstance(history_count, bool) or not isinstance(history_count, int) or history_count < 0:
        raise ValueError("history_count must be a non-negative integer")
    mf, graph, aspect = _validate_components(mf_scores, graph_scores, aspect_scores)
    support_values = _finite_scores(support, "support")
    if set(support_values) != set(mf):
        raise ValueError("support must use the same candidate IDs as hybrid components")
    if any(value < 0.0 or value > 1.0 for value in support_values.values()):
        raise ValueError("support values must be between 0 and 1")
    mf_rank = percentile_normalize(mf)
    graph_rank = percentile_normalize(graph)
    aspect_rank = percentile_normalize(aspect)
    history_factor = kappa / (kappa + history_count)
    return {
        item_id: (
            (1.0 - g_max * history_factor * support_values[item_id])
            * (rho * mf_rank[item_id] + (1.0 - rho) * graph_rank[item_id])
            + g_max * history_factor * support_values[item_id] * aspect_rank[item_id]
        )
        for item_id in mf
    }


adaptive_gate_scores = adaptive_trustrec_scores
trustrec_scores = adaptive_trustrec_scores


@dataclass(frozen=True)
class HybridScore:
    """A score with normalized components and gate details for audit output."""

    item_id: str
    total_score: float
    normalized_mf: float
    normalized_graph: float
    normalized_aspect: float
    mf_contribution: float
    graph_contribution: float
    aspect_contribution: float
    gate: float = 0.0
    support: float = 0.0
    history_count: int = 0

    @property
    def component_scores(self) -> dict[str, float]:
        return {
            "mf": self.mf_contribution,
            "graph": self.graph_contribution,
            "aspect": self.aspect_contribution,
            "normalized_mf": self.normalized_mf,
            "normalized_graph": self.normalized_graph,
            "normalized_aspect": self.normalized_aspect,
            "gate": self.gate,
            "support": self.support,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "total_score": self.total_score,
            "component_scores": self.component_scores,
            "history_count": self.history_count,
        }


def fixed_hybrid_details(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
    *,
    mf_weight: float = 0.5,
    graph_weight: float = 0.25,
    aspect_weight: float = 0.25,
) -> dict[str, HybridScore]:
    """Return fixed-hybrid scores with contribution details."""

    _validate_weights(mf_weight, graph_weight, aspect_weight)
    mf, graph, aspect = _validate_components(mf_scores, graph_scores, aspect_scores)
    normalized = tuple(percentile_normalize(values) for values in (mf, graph, aspect))
    mf_rank, graph_rank, aspect_rank = normalized
    return {
        item_id: HybridScore(
            item_id=item_id,
            total_score=mf_weight * mf_rank[item_id]
            + graph_weight * graph_rank[item_id]
            + aspect_weight * aspect_rank[item_id],
            normalized_mf=mf_rank[item_id],
            normalized_graph=graph_rank[item_id],
            normalized_aspect=aspect_rank[item_id],
            mf_contribution=mf_weight * mf_rank[item_id],
            graph_contribution=graph_weight * graph_rank[item_id],
            aspect_contribution=aspect_weight * aspect_rank[item_id],
        )
        for item_id in mf
    }


def adaptive_trustrec_details(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
    support: Mapping[str, float],
    history_count: int,
    *,
    kappa: float = 5.0,
    g_max: float = 0.5,
    rho: float = 0.5,
) -> dict[str, HybridScore]:
    """Return adaptive scores with per-item gate and score-part details."""

    _validate_gate(kappa, g_max, rho)
    mf, graph, aspect = _validate_components(mf_scores, graph_scores, aspect_scores)
    support_values = _finite_scores(support, "support")
    if set(support_values) != set(mf):
        raise ValueError("support must use the same candidate IDs as hybrid components")
    if any(value < 0.0 or value > 1.0 for value in support_values.values()):
        raise ValueError("support values must be between 0 and 1")
    normalized = tuple(percentile_normalize(values) for values in (mf, graph, aspect))
    mf_rank, graph_rank, aspect_rank = normalized
    history_factor = kappa / (kappa + history_count)
    details: dict[str, HybridScore] = {}
    for item_id in mf:
        gate = g_max * history_factor * support_values[item_id]
        base = rho * mf_rank[item_id] + (1.0 - rho) * graph_rank[item_id]
        details[item_id] = HybridScore(
            item_id=item_id,
            total_score=(1.0 - gate) * base + gate * aspect_rank[item_id],
            normalized_mf=mf_rank[item_id],
            normalized_graph=graph_rank[item_id],
            normalized_aspect=aspect_rank[item_id],
            mf_contribution=(1.0 - gate) * rho * mf_rank[item_id],
            graph_contribution=(1.0 - gate) * (1.0 - rho) * graph_rank[item_id],
            aspect_contribution=gate * aspect_rank[item_id],
            gate=gate,
            support=support_values[item_id],
            history_count=history_count,
        )
    return details


def rank_scores(
    scores: Mapping[str, float], *, k: int | None = None
) -> tuple[tuple[str, float], ...]:
    """Sort scores by descending value with a stable item ID tie-breaker."""

    values = _finite_scores(scores, "scores")
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError("k must be a positive integer or None")
    ranked = sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))
    return tuple(ranked if k is None else ranked[:k])
