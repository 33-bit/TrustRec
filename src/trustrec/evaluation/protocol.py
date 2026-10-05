"""Temporal evaluation cases, ranking validation, slices, and comparison output."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trustrec.recommenders.contracts import (
    Timestamp,
    coerce_interactions,
    first_interactions_before_cutoff,
    stable_model_hash,
    timestamp_value,
)

from .leakage import LeakageError
from .metrics import (
    BootstrapSummary,
    intra_list_diversity,
    ndcg_at_k,
    paired_user_bootstrap,
    precision_at_k,
    recall_at_k,
)

HISTORY_SLICES = ("no_history", "1-2", "3-5", ">5")
POPULARITY_SLICES = ("low", "medium", "high")
EXCLUSION_REASONS = (
    "target_before_cutoff",
    "target_not_positive",
    "target_unknown_item",
    "target_in_history",
    "target_duplicate_pair",
    "target_review_overlap",
    "no_candidates",
)


def _as_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _numeric_rating(row: Mapping[str, Any]) -> float | None:
    value = row.get("rating")
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("rating must be numeric when provided")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("rating must be numeric when provided") from error
    if not math.isfinite(parsed):
        raise ValueError("rating must be finite")
    return parsed


def _history_slice(history_count: int) -> str:
    if history_count == 0:
        return "no_history"
    if history_count <= 2:
        return "1-2"
    if history_count <= 5:
        return "3-5"
    return ">5"


def _popularity_groups(counts: Mapping[str, int]) -> dict[str, str]:
    ordered = sorted(counts, key=lambda item_id: (int(counts[item_id]), item_id))
    if not ordered:
        return {}
    groups: dict[str, str] = {}
    for index, item_id in enumerate(ordered):
        group_index = min(2, index * len(POPULARITY_SLICES) // len(ordered))
        groups[item_id] = POPULARITY_SLICES[group_index]
    return groups


def _timestamp_for_error(value: Any) -> float:
    try:
        return timestamp_value(value)
    except ValueError as error:
        raise ValueError("timestamps must be numeric or timezone-aware ISO values") from error


@dataclass(frozen=True)
class EvaluationCase:
    """One target user with its locked candidate and target sets."""

    user_id: str
    candidate_item_ids: tuple[str, ...]
    target_item_ids: frozenset[str]
    history_count: int
    history_slice: str
    target_popularity_slices: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        _as_non_empty_string(self.user_id, "user_id")
        candidates = tuple(self.candidate_item_ids)
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_item_ids must be unique")
        if any(not isinstance(item_id, str) or not item_id for item_id in candidates):
            raise ValueError("candidate_item_ids must contain non-empty strings")
        targets = frozenset(self.target_item_ids)
        if not targets <= set(candidates):
            raise ValueError("target_item_ids must be candidate items")
        if isinstance(self.history_count, bool) or self.history_count < 0:
            raise ValueError("history_count must be non-negative")
        if self.history_slice not in HISTORY_SLICES:
            raise ValueError("invalid history_slice")
        groups = {
            str(group): frozenset(items) for group, items in self.target_popularity_slices.items()
        }
        if set(groups) - set(POPULARITY_SLICES):
            raise ValueError("invalid target popularity slice")
        if any(not items <= targets for items in groups.values()):
            raise ValueError("target popularity slices must contain target items")
        object.__setattr__(self, "candidate_item_ids", candidates)
        object.__setattr__(self, "target_item_ids", targets)
        object.__setattr__(self, "target_popularity_slices", groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "candidate_item_ids": list(self.candidate_item_ids),
            "target_item_ids": sorted(self.target_item_ids),
            "history_count": self.history_count,
            "history_slice": self.history_slice,
            "target_popularity_slices": {
                group: sorted(items) for group, items in self.target_popularity_slices.items()
            },
        }


@dataclass(frozen=True)
class EvaluationDataset:
    """The immutable user set shared by all compared ranking models."""

    snapshot_id: str
    cutoff_timestamp: Timestamp
    cases: tuple[EvaluationCase, ...]
    exclusions: Mapping[str, int]
    candidate_set_hash: str
    popularity_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        _as_non_empty_string(self.snapshot_id, "snapshot_id")
        timestamp_value(self.cutoff_timestamp)
        cases = tuple(self.cases)
        if len({case.user_id for case in cases}) != len(cases):
            raise ValueError("evaluation cases must use unique user IDs")
        exclusions = {str(reason): int(count) for reason, count in self.exclusions.items()}
        if any(count < 0 for count in exclusions.values()):
            raise ValueError("exclusion counts cannot be negative")
        popularity_counts = {
            str(item_id): int(count) for item_id, count in self.popularity_counts.items()
        }
        if any(count < 0 for count in popularity_counts.values()):
            raise ValueError("popularity counts cannot be negative")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "popularity_counts", popularity_counts)

    def to_dict(self) -> dict[str, Any]:
        cutoff = self.cutoff_timestamp
        if isinstance(cutoff, datetime):
            cutoff = cutoff.isoformat()
        return {
            "snapshot_id": self.snapshot_id,
            "cutoff_timestamp": cutoff,
            "cases": [case.to_dict() for case in self.cases],
            "exclusions": dict(self.exclusions),
            "candidate_set_hash": self.candidate_set_hash,
            "popularity_counts": dict(self.popularity_counts),
        }


@dataclass(frozen=True)
class ModelEvaluation:
    """Metrics and lineage for one model on the shared evaluation cases."""

    model_id: str
    metrics: Mapping[str, float | None]
    bootstrap: Mapping[str, BootstrapSummary]
    slices: Mapping[str, Any]
    counts: Mapping[str, int | float]
    resource: Mapping[str, float | None]
    per_user: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "metrics": dict(self.metrics),
            "bootstrap": {metric: summary.to_dict() for metric, summary in self.bootstrap.items()},
            "slices": dict(self.slices),
            "counts": dict(self.counts),
            "resource": dict(self.resource),
            "per_user": [dict(row) for row in self.per_user],
            "metadata": dict(self.metadata),
        }


def _target_review_ids(rows: Iterable[Mapping[str, Any]]) -> set[str] | None:
    values: set[str] = set()
    found = False
    for row in rows:
        if "review_id" in row and row.get("review_id") not in (None, ""):
            found = True
            values.add(str(row["review_id"]))
    return values if found else None


def build_evaluation_cases(
    interactions: Iterable[Mapping[str, Any]] | Any,
    targets: Iterable[Mapping[str, Any]] | Any,
    *,
    snapshot_id: str,
    cutoff_timestamp: Timestamp,
    positive_threshold: float = 4.0,
) -> EvaluationDataset:
    """Build one locked evaluation case for every eligible target user."""

    _as_non_empty_string(snapshot_id, "snapshot_id")
    cutoff_value = _timestamp_for_error(cutoff_timestamp)
    if not math.isfinite(float(positive_threshold)):
        raise ValueError("positive_threshold must be finite")
    interaction_rows = coerce_interactions(interactions)
    for row in interaction_rows:
        if _timestamp_for_error(row["timestamp"]) >= cutoff_value:
            raise LeakageError("interaction rows must be strictly before the evaluation cutoff")
    past_rows = first_interactions_before_cutoff(interaction_rows, cutoff_timestamp)
    catalog = {row["item_id"] for row in past_rows}
    history_by_user: dict[str, set[str]] = defaultdict(set)
    positive_counts: Counter[str] = Counter()
    for row in past_rows:
        history_by_user[row["user_id"]].add(row["item_id"])
        rating = _numeric_rating(row)
        if rating is not None and rating >= positive_threshold:
            positive_counts[row["item_id"]] += 1
    popularity_groups = _popularity_groups(
        {item_id: positive_counts.get(item_id, 0) for item_id in catalog}
    )

    target_rows = list(targets.to_dict("records") if hasattr(targets, "to_dict") else targets)
    interaction_review_ids = {
        str(row["review_id"]) for row in interaction_rows if row.get("review_id") not in (None, "")
    }
    target_review_ids = _target_review_ids(target_rows)
    if target_review_ids is not None:
        overlap = interaction_review_ids & target_review_ids
        if overlap:
            raise LeakageError("target review IDs overlap interaction rows")

    exclusions = {reason: 0 for reason in EXCLUSION_REASONS}
    target_by_user: dict[str, set[str]] = defaultdict(set)
    for position, row in enumerate(target_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"target row {position} must be a mapping")
        user_id = _as_non_empty_string(row.get("user_id"), "target user_id")
        item_id = _as_non_empty_string(row.get("item_id"), "target item_id")
        if "timestamp" not in row or row.get("timestamp") in (None, ""):
            raise ValueError("target rows require timestamp")
        timestamp = _timestamp_for_error(row["timestamp"])
        if timestamp < cutoff_value:
            exclusions["target_before_cutoff"] += 1
            continue
        rating = _numeric_rating(row)
        if rating is None or rating < positive_threshold:
            exclusions["target_not_positive"] += 1
            continue
        if item_id not in catalog:
            exclusions["target_unknown_item"] += 1
            continue
        if item_id in history_by_user.get(user_id, set()):
            exclusions["target_in_history"] += 1
            continue
        if item_id in target_by_user[user_id]:
            exclusions["target_duplicate_pair"] += 1
            continue
        target_by_user[user_id].add(item_id)

    cases: list[EvaluationCase] = []
    for user_id in sorted(target_by_user):
        history = history_by_user.get(user_id, set())
        candidates = tuple(sorted(catalog - history))
        if not candidates:
            exclusions["no_candidates"] += 1
            continue
        targets_for_user = frozenset(target_by_user[user_id])
        target_slices: dict[str, frozenset[str]] = {}
        for group in POPULARITY_SLICES:
            selected = frozenset(
                item_id for item_id in targets_for_user if popularity_groups.get(item_id) == group
            )
            if selected:
                target_slices[group] = selected
        cases.append(
            EvaluationCase(
                user_id=user_id,
                candidate_item_ids=candidates,
                target_item_ids=targets_for_user,
                history_count=len(history),
                history_slice=_history_slice(len(history)),
                target_popularity_slices=target_slices,
            )
        )
    candidate_set_hash = stable_model_hash(
        {
            "snapshot_id": snapshot_id,
            "cutoff_timestamp": cutoff_value,
            "cases": [case.to_dict() for case in cases],
        }
    )
    return EvaluationDataset(
        snapshot_id=snapshot_id,
        cutoff_timestamp=cutoff_timestamp,
        cases=tuple(cases),
        exclusions=exclusions,
        candidate_set_hash=candidate_set_hash,
        popularity_counts=dict(positive_counts),
    )


def _mapping_rows(rows: Iterable[Mapping[str, Any]] | Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    result: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"ranking row {position} must be a mapping")
        copied = dict(row)
        if "ranked_item_ids" not in copied and isinstance(copied.get("items"), list):
            items = [item for item in copied["items"] if isinstance(item, Mapping)]
            items.sort(key=lambda item: int(item.get("rank", 0)))
            copied["ranked_item_ids"] = [item.get("item_id") for item in items]
        result.append(copied)
    return result


def _validate_optional_number(
    value: Any, field: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{field} is outside its allowed range")
    return parsed


def _validate_model_rows(
    dataset: EvaluationDataset,
    model_id: str,
    rows: Iterable[Mapping[str, Any]] | Any,
) -> dict[str, dict[str, Any]]:
    _as_non_empty_string(model_id, "model_id")
    cases = {case.user_id: case for case in dataset.cases}
    validated: dict[str, dict[str, Any]] = {}
    for position, raw_row in enumerate(_mapping_rows(rows)):
        user_id = _as_non_empty_string(raw_row.get("user_id"), "ranking user_id")
        if user_id in validated:
            raise LeakageError(f"{model_id} contains duplicate user rows")
        case = cases.get(user_id)
        if case is None:
            raise LeakageError(f"{model_id} contains an unknown evaluation user")
        if raw_row.get("snapshot_id") != dataset.snapshot_id:
            raise LeakageError(f"{model_id} snapshot does not match evaluation dataset")
        if "cutoff_timestamp" not in raw_row:
            raise ValueError(f"{model_id} ranking row {position} lacks cutoff_timestamp")
        if timestamp_value(raw_row["cutoff_timestamp"]) != timestamp_value(
            dataset.cutoff_timestamp
        ):
            raise LeakageError(f"{model_id} cutoff does not match evaluation dataset")
        candidate_ids = raw_row.get("candidate_item_ids")
        ranked_ids = raw_row.get("ranked_item_ids")
        if not isinstance(candidate_ids, (list, tuple)):
            raise ValueError(f"{model_id} candidate_item_ids must be a list")
        if not isinstance(ranked_ids, (list, tuple)):
            raise ValueError(f"{model_id} ranked_item_ids must be a list")
        candidate_ids = tuple(candidate_ids)
        ranked_ids = tuple(ranked_ids)
        if set(candidate_ids) != set(case.candidate_item_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise LeakageError(f"{model_id} candidate set does not match evaluation dataset")
        if len(ranked_ids) != len(set(ranked_ids)):
            raise LeakageError(f"{model_id} ranked items contain duplicates")
        if not set(ranked_ids) <= set(candidate_ids):
            raise LeakageError(f"{model_id} ranks an item outside its candidate set")
        declared_targets = raw_row.get("target_item_ids")
        if declared_targets is not None and set(declared_targets) != set(case.target_item_ids):
            raise LeakageError(f"{model_id} target set does not match evaluation dataset")
        explanation = _validate_optional_number(
            raw_row.get("explanation_coverage"), "explanation_coverage", minimum=0.0
        )
        if explanation is not None and explanation > 1.0:
            raise ValueError("explanation_coverage is outside its allowed range")
        latency = _validate_optional_number(raw_row.get("latency_ms"), "latency_ms", minimum=0.0)
        fit_time = _validate_optional_number(raw_row.get("fit_time_ms"), "fit_time_ms", minimum=0.0)
        memory = _validate_optional_number(raw_row.get("memory_bytes"), "memory_bytes", minimum=0.0)
        validated[user_id] = {
            "user_id": user_id,
            "candidate_item_ids": list(candidate_ids),
            "ranked_item_ids": list(ranked_ids),
            "explanation_coverage": explanation,
            "latency_ms": latency,
            "fit_time_ms": fit_time,
            "memory_bytes": memory,
            "metadata": dict(raw_row.get("metadata", {}))
            if isinstance(raw_row.get("metadata"), Mapping)
            else {},
        }
    missing = sorted(set(cases) - set(validated))
    if missing:
        raise LeakageError(f"{model_id} ranking rows lack users: {', '.join(missing[:3])}")
    return validated


def _mean(values: Iterable[float]) -> float | None:
    values_list = [float(value) for value in values]
    return float(sum(values_list) / len(values_list)) if values_list else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    values_list = sorted(float(value) for value in values)
    if not values_list:
        return None
    if len(values_list) == 1:
        return values_list[0]
    position = (len(values_list) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values_list) - 1)
    weight = position - lower
    return float(values_list[lower] * (1.0 - weight) + values_list[upper] * weight)


def _metric_row(
    case: EvaluationCase,
    ranking: Mapping[str, Any],
    *,
    target_item_ids: Collection[str],
    k_values: Sequence[int],
    item_aspects: Mapping[str, Collection[str]] | None,
) -> dict[str, Any]:
    ranked = ranking["ranked_item_ids"]
    result: dict[str, Any] = {"user_id": case.user_id}
    for k in k_values:
        result[f"ndcg@{k}"] = ndcg_at_k(ranked, target_item_ids, k)
        result[f"recall@{k}"] = recall_at_k(ranked, target_item_ids, k)
        result[f"precision@{k}"] = precision_at_k(ranked, target_item_ids, k)
        result[f"diversity@{k}"] = (
            intra_list_diversity(ranked, item_aspects, k) if item_aspects is not None else None
        )
    result["coverage"] = 1.0 if ranked else 0.0
    result["candidate_retention"] = (
        len(ranked) / len(case.candidate_item_ids) if case.candidate_item_ids else 0.0
    )
    result["explanation_coverage"] = ranking.get("explanation_coverage")
    result["latency_ms"] = ranking.get("latency_ms")
    result["fit_time_ms"] = ranking.get("fit_time_ms")
    result["memory_bytes"] = ranking.get("memory_bytes")
    return result


def _aggregate_metric_rows(
    rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> dict[str, float | None]:
    return {
        metric: _mean(row[metric] for row in rows if row.get(metric) is not None)
        for metric in metric_names
    }


def _slice_payload(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    target_count: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "counts": {"users": len(rows)},
        "metrics": _aggregate_metric_rows(rows, metric_names),
    }
    if target_count is not None:
        payload["counts"]["targets"] = target_count
    return payload


def _bootstrap_for_metric(
    metric: str,
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    user_order: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence_level: float,
    reference_model: str | None,
) -> dict[str, BootstrapSummary]:
    available_users = [
        user_id
        for user_id in user_order
        if all(
            model_id in model_rows
            and any(row.get("user_id") == user_id and row.get(metric) is not None for row in rows)
            for model_id, rows in model_rows.items()
        )
    ]
    if not available_users:
        return {}
    row_maps = {
        model_id: {str(row["user_id"]): row for row in rows}
        for model_id, rows in model_rows.items()
    }
    values = {
        model_id: [float(row_maps[model_id][user_id][metric]) for user_id in available_users]
        for model_id in model_rows
    }
    return paired_user_bootstrap(
        values,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        reference_model=reference_model,
    )


def compare_model_rankings(
    dataset: EvaluationDataset,
    model_rows: Mapping[str, Iterable[Mapping[str, Any]] | Any],
    *,
    k_values: Sequence[int] = (5, 10, 20),
    item_aspects: Mapping[str, Collection[str]] | None = None,
    bootstrap_seed: int = 7,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    reference_model: str | None = None,
) -> dict[str, ModelEvaluation]:
    """Validate and compare model rankings on one shared evaluation dataset."""

    if not model_rows:
        raise ValueError("model_rows cannot be empty")
    normalized_k = tuple(k_values)
    if not normalized_k or len(set(normalized_k)) != len(normalized_k):
        raise ValueError("k_values must contain unique positive integers")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in normalized_k):
        raise ValueError("k_values must contain unique positive integers")
    if reference_model is not None and reference_model not in model_rows:
        raise ValueError("reference_model is not present in model_rows")

    validated = {
        model_id: _validate_model_rows(dataset, model_id, rows)
        for model_id, rows in model_rows.items()
    }
    user_order = [case.user_id for case in dataset.cases]
    metric_names = [
        metric
        for k in normalized_k
        for metric in (f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"diversity@{k}")
    ] + ["coverage", "candidate_retention", "explanation_coverage"]
    model_metric_rows: dict[str, list[dict[str, Any]]] = {}
    model_results: dict[str, ModelEvaluation] = {}
    for model_id, ranking_by_user in validated.items():
        rows = [
            _metric_row(
                case,
                ranking_by_user[case.user_id],
                target_item_ids=case.target_item_ids,
                k_values=normalized_k,
                item_aspects=item_aspects,
            )
            for case in dataset.cases
        ]
        model_metric_rows[model_id] = rows
        metrics = _aggregate_metric_rows(rows, metric_names)
        counts: dict[str, int | float] = {
            "eligible_users": len(dataset.cases),
            "evaluated_users": len(rows),
            "target_items": sum(len(case.target_item_ids) for case in dataset.cases),
            "candidate_items": sum(len(case.candidate_item_ids) for case in dataset.cases),
            "ranked_users": sum(1 for row in rows if row["coverage"] > 0),
            "explanation_users": sum(
                1 for row in rows if row.get("explanation_coverage") is not None
            ),
        }
        resource = {
            "latency_ms_mean": _mean(
                row["latency_ms"] for row in rows if row.get("latency_ms") is not None
            ),
            "latency_ms_p95": _percentile(
                (row["latency_ms"] for row in rows if row.get("latency_ms") is not None), 0.95
            ),
            "fit_time_ms_mean": _mean(
                row["fit_time_ms"] for row in rows if row.get("fit_time_ms") is not None
            ),
            "memory_bytes_mean": _mean(
                row["memory_bytes"] for row in rows if row.get("memory_bytes") is not None
            ),
        }
        history_slices: dict[str, Any] = {}
        for group in HISTORY_SLICES:
            selected = [
                row
                for case, row in zip(dataset.cases, rows, strict=True)
                if case.history_slice == group
            ]
            history_slices[group] = _slice_payload(selected, metric_names)
        popularity_slices: dict[str, Any] = {}
        for group in POPULARITY_SLICES:
            selected_rows: list[dict[str, Any]] = []
            target_count = 0
            for case, _ranking in zip(dataset.cases, rows, strict=True):
                target_ids = case.target_popularity_slices.get(group, frozenset())
                if not target_ids:
                    continue
                selected_rows.append(
                    _metric_row(
                        case,
                        validated[model_id][case.user_id],
                        target_item_ids=target_ids,
                        k_values=normalized_k,
                        item_aspects=item_aspects,
                    )
                )
                target_count += len(target_ids)
            popularity_slices[group] = _slice_payload(
                selected_rows,
                metric_names,
                target_count=target_count,
            )
        model_results[model_id] = ModelEvaluation(
            model_id=model_id,
            metrics=metrics,
            bootstrap={},
            slices={
                "history": history_slices,
                "popularity": popularity_slices,
                "exclusions": dict(dataset.exclusions),
            },
            counts=counts,
            resource=resource,
            per_user=tuple(rows),
            metadata={"candidate_set_hash": dataset.candidate_set_hash},
        )

    bootstrap_by_model: dict[str, dict[str, BootstrapSummary]] = defaultdict(dict)
    for metric in metric_names:
        summaries = _bootstrap_for_metric(
            metric,
            model_metric_rows,
            user_order,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            reference_model=reference_model,
        )
        for model_id, summary in summaries.items():
            bootstrap_by_model[model_id][metric] = summary
    return {
        model_id: ModelEvaluation(
            model_id=result.model_id,
            metrics=result.metrics,
            bootstrap=bootstrap_by_model.get(model_id, {}),
            slices=result.slices,
            counts=result.counts,
            resource=result.resource,
            per_user=result.per_user,
            metadata=result.metadata,
        )
        for model_id, result in model_results.items()
    }
