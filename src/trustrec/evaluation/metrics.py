"""Pure recommendation metrics and paired user bootstrap summaries."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _ranked_prefix(ranked_item_ids: Sequence[str], k: int) -> tuple[str, ...]:
    _validate_k(k)
    ranked = tuple(ranked_item_ids)
    if any(not isinstance(item_id, str) or not item_id for item_id in ranked):
        raise ValueError("ranked item IDs must be non-empty strings")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked item IDs cannot contain duplicates")
    return ranked[:k]


def _target_set(target_item_ids: Collection[str]) -> frozenset[str]:
    targets = frozenset(target_item_ids)
    if any(not isinstance(item_id, str) or not item_id for item_id in targets):
        raise ValueError("target item IDs must be non-empty strings")
    return targets


def ndcg_at_k(ranked_item_ids: Sequence[str], target_item_ids: Collection[str], k: int) -> float:
    """Return binary NDCG for one ranked list and one target set."""

    ranked = _ranked_prefix(ranked_item_ids, k)
    targets = _target_set(target_item_ids)
    if not targets:
        return 0.0
    dcg = sum(
        (1.0 if item_id in targets else 0.0) / math.log2(rank + 2)
        for rank, item_id in enumerate(ranked)
    )
    ideal_length = min(k, len(targets))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_length))
    return float(dcg / ideal) if ideal else 0.0


def recall_at_k(ranked_item_ids: Sequence[str], target_item_ids: Collection[str], k: int) -> float:
    """Return the fraction of targets found in the first K ranks."""

    ranked = _ranked_prefix(ranked_item_ids, k)
    targets = _target_set(target_item_ids)
    if not targets:
        return 0.0
    return float(len(set(ranked) & targets) / len(targets))


def precision_at_k(
    ranked_item_ids: Sequence[str], target_item_ids: Collection[str], k: int
) -> float:
    """Return binary precision using K as the denominator."""

    ranked = _ranked_prefix(ranked_item_ids, k)
    targets = _target_set(target_item_ids)
    if not targets:
        return 0.0
    return float(len(set(ranked) & targets) / k)


def intra_list_diversity(
    ranked_item_ids: Sequence[str], item_aspects: Mapping[str, Collection[str]], k: int
) -> float | None:
    """Return mean pairwise aspect Jaccard distance for the first K items."""

    ranked = _ranked_prefix(ranked_item_ids, k)
    aspect_sets = [set(item_aspects[item_id]) for item_id in ranked if item_id in item_aspects]
    if not aspect_sets:
        return None
    if len(aspect_sets) < 2:
        return 0.0
    distances: list[float] = []
    for left, right in combinations(aspect_sets, 2):
        union = left | right
        similarity = len(left & right) / len(union) if union else 1.0
        distances.append(1.0 - similarity)
    return float(sum(distances) / len(distances))


@dataclass(frozen=True)
class BootstrapSummary:
    """A percentile interval for a user-level metric."""

    estimate: float
    lower: float
    upper: float
    samples: int
    seed: int
    confidence_level: float
    delta_estimate: float | None = None
    delta_lower: float | None = None
    delta_upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "samples": self.samples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "delta_estimate": self.delta_estimate,
            "delta_lower": self.delta_lower,
            "delta_upper": self.delta_upper,
        }


def _validate_bootstrap_inputs(
    values_by_model: Mapping[str, Sequence[float]],
    *,
    seed: int,
    samples: int,
    confidence_level: float,
    reference_model: str | None,
) -> tuple[dict[str, np.ndarray], int]:
    if not values_by_model:
        raise ValueError("values_by_model cannot be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not math.isfinite(float(confidence_level)) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if reference_model is not None and reference_model not in values_by_model:
        raise ValueError("reference_model is not present in values_by_model")
    arrays: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for model_id, values in values_by_model.items():
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model IDs must be non-empty strings")
        array = np.asarray(tuple(values), dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("each model must contain at least one user value")
        if not np.isfinite(array).all():
            raise ValueError("bootstrap values must be finite")
        if expected_length is None:
            expected_length = int(array.size)
        elif array.size != expected_length:
            raise ValueError("all models must have the same user count")
        arrays[model_id] = array
    return arrays, expected_length or 0


def paired_user_bootstrap(
    values_by_model: Mapping[str, Sequence[float]],
    *,
    seed: int,
    samples: int,
    confidence_level: float,
    reference_model: str | None = None,
) -> dict[str, BootstrapSummary]:
    """Compute paired user bootstrap intervals for one or more models."""

    arrays, _ = _validate_bootstrap_inputs(
        values_by_model,
        seed=seed,
        samples=samples,
        confidence_level=confidence_level,
        reference_model=reference_model,
    )
    alpha = (1.0 - confidence_level) / 2.0
    rng = np.random.default_rng(seed)
    user_count = next(iter(arrays.values())).size
    batch_size = 64
    replicate_values = {model_id: np.empty(samples, dtype=np.float64) for model_id in arrays}
    replicate_deltas: dict[str, np.ndarray] = {}
    reference_values = arrays.get(reference_model) if reference_model else None
    if reference_values is not None:
        replicate_deltas = {
            model_id: np.empty(samples, dtype=np.float64)
            for model_id in arrays
            if model_id != reference_model
        }
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, user_count, size=(stop - start, user_count))
        for model_id, values in arrays.items():
            replicate_values[model_id][start:stop] = values[indices].mean(axis=1)
            if reference_values is not None and model_id != reference_model:
                replicate_deltas[model_id][start:stop] = (
                    values[indices] - reference_values[indices]
                ).mean(axis=1)

    summaries: dict[str, BootstrapSummary] = {}
    for model_id, values in arrays.items():
        estimate = float(values.mean())
        distribution = replicate_values[model_id]
        lower = float(np.quantile(distribution, alpha))
        upper = float(np.quantile(distribution, 1.0 - alpha))
        if reference_values is None or model_id == reference_model:
            summaries[model_id] = BootstrapSummary(
                estimate=estimate,
                lower=lower,
                upper=upper,
                samples=samples,
                seed=seed,
                confidence_level=confidence_level,
            )
            continue
        deltas = replicate_deltas[model_id]
        delta = float((values - reference_values).mean())
        summaries[model_id] = BootstrapSummary(
            estimate=estimate,
            lower=lower,
            upper=upper,
            samples=samples,
            seed=seed,
            confidence_level=confidence_level,
            delta_estimate=delta,
            delta_lower=float(np.quantile(deltas, alpha)),
            delta_upper=float(np.quantile(deltas, 1.0 - alpha)),
        )
    return summaries
