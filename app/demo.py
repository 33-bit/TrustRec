"""Prepared-artifact service for the TrustRec Streamlit demo."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE_PATH = Path(__file__).with_name("demo_bundle.json")
SUPPORT_THRESHOLD = 0.25
SUPPORTED_SCHEMA_VERSION = 1


class DemoValidationError(ValueError):
    """Raised when a prepared demo bundle does not meet the serving contract."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoValidationError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise DemoValidationError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DemoValidationError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise DemoValidationError(f"{field} must be finite")
    return number


def _timestamp_text(value: Any, field: str) -> str:
    timestamp = _required_text(value, field)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise DemoValidationError(f"{field} must be an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise DemoValidationError(f"{field} must include a timezone")
    return timestamp


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DemoValidationError(f"{field} must be an object")
    return value


def _number_map(value: Any, field: str) -> dict[str, float]:
    source = _mapping(value, field)
    result: dict[str, float] = {}
    for key, raw_value in source.items():
        name = _required_text(key, f"{field} key")
        result[name] = _finite_number(raw_value, f"{field}.{name}")
    return result


def _non_negative_number_map(value: Mapping[str, Any], field: str) -> dict[str, float]:
    result = _number_map(value, field)
    if any(number < 0.0 for number in result.values()):
        raise DemoValidationError(f"{field} values must be non-negative")
    return result


@dataclass(frozen=True)
class DemoUser:
    """One prepared user and learned aspect preferences."""

    user_id: str
    history_count: int
    history_item_ids: tuple[str, ...]
    learned_aspect_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        _required_text(self.user_id, "user_id")
        if isinstance(self.history_count, bool) or not isinstance(self.history_count, int):
            raise DemoValidationError("history_count must be an integer")
        if self.history_count < 0:
            raise DemoValidationError("history_count must be non-negative")
        if len(self.history_item_ids) != len(set(self.history_item_ids)):
            raise DemoValidationError("history_item_ids must be unique")
        for item_id in self.history_item_ids:
            _required_text(item_id, "history item_id")
        weights = _non_negative_number_map(self.learned_aspect_weights, "learned_aspect_weights")
        if not weights or sum(weights.values()) <= 0.0:
            raise DemoValidationError("learned_aspect_weights must contain positive mass")
        object.__setattr__(self, "learned_aspect_weights", weights)


@dataclass(frozen=True)
class DemoItem:
    """One item with aspect scores used by the temporary priority control."""

    item_id: str
    title: str
    key_aspects: tuple[str, ...]
    aspect_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        _required_text(self.item_id, "item_id")
        _required_text(self.title, "title")
        if len(self.key_aspects) != len(set(self.key_aspects)):
            raise DemoValidationError("key_aspects must be unique")
        for aspect in self.key_aspects:
            _required_text(aspect, "key aspect")
        scores = _number_map(self.aspect_scores, "aspect_scores")
        if any(score < 0.0 or score > 1.0 for score in scores.values()):
            raise DemoValidationError("aspect_scores must be between 0 and 1")
        if not scores:
            raise DemoValidationError("aspect_scores must not be empty")
        object.__setattr__(self, "aspect_scores", scores)


@dataclass(frozen=True)
class DemoRanking:
    """One prepared model score for one candidate item."""

    item_id: str
    rank: int
    total_score: float
    component_scores: Mapping[str, float]
    aspect_weight: float = 0.0
    support: float = 0.0

    def __post_init__(self) -> None:
        _required_text(self.item_id, "ranking item_id")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise DemoValidationError("ranking rank must be a positive integer")
        total_score = _finite_number(self.total_score, "ranking total_score")
        component_scores = _number_map(self.component_scores, "component_scores")
        aspect_weight = _finite_number(self.aspect_weight, "aspect_weight")
        support = _finite_number(self.support, "support")
        if not 0.0 <= aspect_weight <= 1.0:
            raise DemoValidationError("aspect_weight must be between 0 and 1")
        if not 0.0 <= support <= 1.0:
            raise DemoValidationError("support must be between 0 and 1")
        object.__setattr__(self, "total_score", total_score)
        object.__setattr__(self, "component_scores", component_scores)
        object.__setattr__(self, "aspect_weight", aspect_weight)
        object.__setattr__(self, "support", support)


@dataclass(frozen=True)
class DemoEvidence:
    """One source passage that can be shown without reviewer identity."""

    item_id: str
    aspect: str
    review_id: str
    sentiment: str
    support: float
    text: str
    timestamp: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.item_id, "evidence item_id"),
            (self.aspect, "evidence aspect"),
            (self.review_id, "evidence review_id"),
            (self.text, "evidence text"),
        ):
            _required_text(value, field)
        if self.sentiment not in {"positive", "negative", "neutral"}:
            raise DemoValidationError("evidence sentiment must be positive, negative, or neutral")
        support = _finite_number(self.support, "evidence support")
        if not 0.0 <= support <= 1.0:
            raise DemoValidationError("evidence support must be between 0 and 1")
        timestamp = _timestamp_text(self.timestamp, "evidence timestamp")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "timestamp", timestamp)

    def to_display(self) -> dict[str, Any]:
        """Return fields that are safe for the presenter view."""

        return {
            "aspect": self.aspect,
            "review_id": self.review_id,
            "sentiment": self.sentiment,
            "support": self.support,
            "text": self.text,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class DemoMetric:
    """One prepared metric row shown by the demo."""

    metric: str
    value: float
    split_role: str
    run_id: str
    model_id: str
    model_hash: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.metric, "metric"),
            (self.split_role, "split_role"),
            (self.run_id, "run_id"),
            (self.model_id, "model_id"),
            (self.model_hash, "model_hash"),
        ):
            _required_text(value, field)
        object.__setattr__(self, "value", _finite_number(self.value, "metric value"))


@dataclass(frozen=True)
class DemoBundle:
    """Validated records used by one demo process."""

    schema_version: int
    snapshot_id: str
    dataset_hash: str
    cutoff_timestamp: str
    configuration_hash: str
    model_hashes: Mapping[str, str]
    users: Mapping[str, DemoUser]
    items: Mapping[str, DemoItem]
    rankings: Mapping[str, Mapping[str, tuple[DemoRanking, ...]]]
    evidence: tuple[DemoEvidence, ...]
    metrics: tuple[DemoMetric, ...]


@dataclass(frozen=True)
class DemoLoadResult:
    """Bundle plus fallback information for the Streamlit view."""

    bundle: DemoBundle
    used_fallback: bool
    load_error: str | None = None


@dataclass(frozen=True)
class DemoRecommendation:
    """One ranked item with display-ready score and explanation fields."""

    model_id: str
    item_id: str
    title: str
    rank: int
    total_score: float
    score_parts: Mapping[str, float]
    aspect_scores: Mapping[str, float]
    support: float
    learned_aspect_weights: Mapping[str, float]
    temporary_priorities: Mapping[str, float]
    effective_aspect_weights: Mapping[str, float]
    evidence_state: str


def _parse_user(source: Mapping[str, Any]) -> DemoUser:
    history = source.get("history_item_ids", [])
    if not isinstance(history, list):
        raise DemoValidationError("history_item_ids must be a list")
    return DemoUser(
        user_id=_required_text(source.get("user_id"), "user_id"),
        history_count=source.get("history_count"),
        history_item_ids=tuple(_required_text(value, "history item_id") for value in history),
        learned_aspect_weights=_number_map(
            source.get("learned_aspect_weights"), "learned_aspect_weights"
        ),
    )


def _parse_item(source: Mapping[str, Any]) -> DemoItem:
    key_aspects = source.get("key_aspects", [])
    if not isinstance(key_aspects, list):
        raise DemoValidationError("key_aspects must be a list")
    return DemoItem(
        item_id=_required_text(source.get("item_id"), "item_id"),
        title=_required_text(source.get("title"), "title"),
        key_aspects=tuple(_required_text(value, "key aspect") for value in key_aspects),
        aspect_scores=_number_map(source.get("aspect_scores"), "aspect_scores"),
    )


def _parse_ranking(source: Mapping[str, Any]) -> DemoRanking:
    return DemoRanking(
        item_id=_required_text(source.get("item_id"), "ranking item_id"),
        rank=source.get("rank"),
        total_score=source.get("total_score"),
        component_scores=_number_map(source.get("component_scores"), "component_scores"),
        aspect_weight=source.get("aspect_weight", 0.0),
        support=source.get("support", 0.0),
    )


def _parse_evidence(source: Mapping[str, Any]) -> DemoEvidence:
    return DemoEvidence(
        item_id=_required_text(source.get("item_id"), "evidence item_id"),
        aspect=_required_text(source.get("aspect"), "evidence aspect"),
        review_id=_required_text(source.get("review_id"), "evidence review_id"),
        sentiment=_required_text(source.get("sentiment"), "evidence sentiment"),
        support=source.get("support"),
        text=_required_text(source.get("text"), "evidence text"),
        timestamp=_timestamp_text(source.get("timestamp"), "evidence timestamp"),
    )


def _parse_metric(source: Mapping[str, Any]) -> DemoMetric:
    return DemoMetric(
        metric=_required_text(source.get("metric"), "metric"),
        value=source.get("value"),
        split_role=_required_text(source.get("split_role"), "split_role"),
        run_id=_required_text(source.get("run_id"), "run_id"),
        model_id=_required_text(source.get("model_id"), "model_id"),
        model_hash=_required_text(source.get("model_hash"), "model_hash"),
    )


def _parse_bundle(payload: Mapping[str, Any]) -> DemoBundle:
    schema_version = payload.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise DemoValidationError(f"schema_version must be {SUPPORTED_SCHEMA_VERSION}")
    snapshot_id = _required_text(payload.get("snapshot_id"), "snapshot_id")
    dataset_hash = _required_text(payload.get("dataset_hash"), "dataset_hash")
    cutoff_timestamp = _timestamp_text(payload.get("cutoff_timestamp"), "cutoff_timestamp")
    configuration_hash = _required_text(payload.get("configuration_hash"), "configuration_hash")

    model_hashes_source = _mapping(payload.get("model_hashes"), "model_hashes")
    model_hashes = {
        _required_text(model_id, "model hash key"): _required_text(model_hash, "model hash")
        for model_id, model_hash in model_hashes_source.items()
    }
    if not model_hashes:
        raise DemoValidationError("model_hashes must not be empty")

    users_source = payload.get("users")
    if not isinstance(users_source, list) or not users_source:
        raise DemoValidationError("users must be a non-empty list")
    users: dict[str, DemoUser] = {}
    for raw_user in users_source:
        user = _parse_user(_mapping(raw_user, "user"))
        if user.user_id in users:
            raise DemoValidationError("users must have unique user_id values")
        users[user.user_id] = user

    items_source = payload.get("items")
    if not isinstance(items_source, list) or not items_source:
        raise DemoValidationError("items must be a non-empty list")
    items: dict[str, DemoItem] = {}
    for raw_item in items_source:
        item = _parse_item(_mapping(raw_item, "item"))
        if item.item_id in items:
            raise DemoValidationError("items must have unique item_id values")
        items[item.item_id] = item

    rankings_source = _mapping(payload.get("rankings"), "rankings")
    rankings: dict[str, dict[str, tuple[DemoRanking, ...]]] = {}
    for raw_model_id, raw_model_users in rankings_source.items():
        model_id = _required_text(raw_model_id, "ranking model_id")
        model_users_source = _mapping(raw_model_users, f"rankings.{model_id}")
        model_users: dict[str, tuple[DemoRanking, ...]] = {}
        for raw_user_id, raw_rows in model_users_source.items():
            user_id = _required_text(raw_user_id, "ranking user_id")
            if user_id not in users:
                raise DemoValidationError(f"ranking user_id is unknown: {user_id}")
            if not isinstance(raw_rows, list) or not raw_rows:
                raise DemoValidationError(f"rankings.{model_id}.{user_id} must be a non-empty list")
            rows = tuple(_parse_ranking(_mapping(row, "ranking row")) for row in raw_rows)
            item_ids = [row.item_id for row in rows]
            if len(item_ids) != len(set(item_ids)):
                raise DemoValidationError("ranking item IDs must be unique")
            if any(item_id not in items for item_id in item_ids):
                raise DemoValidationError("ranking item_id is not present in items")
            ranks = sorted(row.rank for row in rows)
            if ranks != list(range(1, len(rows) + 1)):
                raise DemoValidationError("ranking ranks must be contiguous and one-based")
            model_users[user_id] = rows
        rankings[model_id] = model_users
    if not rankings:
        raise DemoValidationError("rankings must not be empty")
    unknown_models = set(rankings) - set(model_hashes)
    if unknown_models:
        raise DemoValidationError("model_hashes is missing a ranking model")

    evidence_source = payload.get("evidence", [])
    if not isinstance(evidence_source, list):
        raise DemoValidationError("evidence must be a list")
    evidence = tuple(_parse_evidence(_mapping(row, "evidence row")) for row in evidence_source)
    if any(row.item_id not in items for row in evidence):
        raise DemoValidationError("evidence item_id is not present in items")

    metrics_source = payload.get("metrics", [])
    if not isinstance(metrics_source, list):
        raise DemoValidationError("metrics must be a list")
    metrics = tuple(_parse_metric(_mapping(row, "metric row")) for row in metrics_source)

    return DemoBundle(
        schema_version=schema_version,
        snapshot_id=snapshot_id,
        dataset_hash=dataset_hash,
        cutoff_timestamp=cutoff_timestamp,
        configuration_hash=configuration_hash,
        model_hashes=model_hashes,
        users=users,
        items=items,
        rankings=rankings,
        evidence=evidence,
        metrics=metrics,
    )


def load_demo_bundle(path: Path) -> DemoBundle:
    """Read and validate one prepared demo bundle."""

    bundle_path = Path(path)
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DemoValidationError(f"demo bundle does not exist: {bundle_path}") from error
    except OSError as error:
        raise DemoValidationError(f"demo bundle cannot be read: {bundle_path}") from error
    except json.JSONDecodeError as error:
        raise DemoValidationError(f"demo bundle is not valid JSON: {bundle_path}") from error
    return _parse_bundle(_mapping(payload, "demo bundle"))


def load_demo_bundle_with_fallback(configured_path: Path | None = None) -> DemoLoadResult:
    """Load a configured bundle and use the checked-in fixture after a load error."""

    configured_value = configured_path
    if configured_value is None:
        environment_value = os.environ.get("TRUSTREC_DEMO_BUNDLE", "").strip()
        configured_value = Path(environment_value) if environment_value else None
    if configured_value is None:
        return DemoLoadResult(
            bundle=load_demo_bundle(DEFAULT_BUNDLE_PATH),
            used_fallback=True,
            load_error="no external demo bundle configured",
        )
    try:
        return DemoLoadResult(bundle=load_demo_bundle(configured_value), used_fallback=False)
    except DemoValidationError as error:
        fallback = load_demo_bundle(DEFAULT_BUNDLE_PATH)
        return DemoLoadResult(bundle=fallback, used_fallback=True, load_error=str(error))


def evidence_for_item(bundle: DemoBundle, item_id: str) -> tuple[DemoEvidence, ...]:
    """Return source passages for one item in stable order."""

    if item_id not in bundle.items:
        raise DemoValidationError(f"unknown item_id: {item_id}")
    return tuple(
        sorted(
            (row for row in bundle.evidence if row.item_id == item_id),
            key=lambda row: (row.aspect, row.timestamp, row.review_id),
        )
    )


def evidence_state(rows: Sequence[DemoEvidence]) -> str:
    """Return the support state that the view must show."""

    if not rows:
        return "unavailable"
    sentiments = {row.sentiment for row in rows if row.support >= SUPPORT_THRESHOLD}
    if {"positive", "negative"} <= sentiments:
        return "conflicting"
    if max(row.support for row in rows) < SUPPORT_THRESHOLD:
        return "insufficient_support"
    return "supported"


def _effective_weights(
    learned: Mapping[str, float], priorities: Mapping[str, float]
) -> dict[str, float]:
    combined = dict(learned)
    for aspect, value in priorities.items():
        combined[aspect] = combined.get(aspect, 0.0) + value
    total = sum(combined.values())
    if total <= 0.0:
        return {aspect: 1.0 / len(combined) for aspect in combined}
    return {aspect: value / total for aspect, value in combined.items()}


def _weighted_aspect_score(
    aspect_scores: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    relevant = {aspect: weight for aspect, weight in weights.items() if aspect in aspect_scores}
    total_weight = sum(relevant.values())
    if total_weight <= 0.0:
        return sum(aspect_scores.values()) / len(aspect_scores)
    return sum(aspect_scores[aspect] * weight for aspect, weight in relevant.items()) / total_weight


def _validated_priorities(
    priorities: Mapping[str, float] | None, known_aspects: set[str]
) -> dict[str, float]:
    if priorities is None:
        return {}
    if not isinstance(priorities, Mapping):
        raise DemoValidationError("priorities must be an object")
    result: dict[str, float] = {}
    for raw_aspect, raw_value in priorities.items():
        aspect = _required_text(raw_aspect, "priority aspect")
        if aspect not in known_aspects:
            raise DemoValidationError(f"unknown priority aspect: {aspect}")
        value = _finite_number(raw_value, f"priority.{aspect}")
        if value < 0.0:
            raise DemoValidationError("priority values must be non-negative")
        result[aspect] = value
    return result


def rank_recommendations(
    bundle: DemoBundle,
    *,
    user_id: str,
    model_id: str,
    k: int,
    priorities: Mapping[str, float] | None = None,
) -> tuple[DemoRecommendation, ...]:
    """Return prepared recommendations with optional temporary priorities."""

    if user_id not in bundle.users:
        raise DemoValidationError(f"unknown user_id: {user_id}")
    if model_id not in bundle.rankings:
        raise DemoValidationError(f"unknown model_id: {model_id}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise DemoValidationError("k must be a positive integer")
    model_rows = bundle.rankings[model_id].get(user_id)
    if model_rows is None:
        raise DemoValidationError(f"model has no ranking for user_id: {user_id}")
    user = bundle.users[user_id]
    known_aspects = set(user.learned_aspect_weights)
    for item in bundle.items.values():
        known_aspects.update(item.aspect_scores)
    validated_priorities = _validated_priorities(priorities, known_aspects)
    effective_weights = _effective_weights(user.learned_aspect_weights, validated_priorities)
    recommendations: list[DemoRecommendation] = []
    for row in model_rows:
        item = bundle.items[row.item_id]
        score_parts = dict(row.component_scores)
        total_score = row.total_score
        if model_id == "t0_trustrec" and row.aspect_weight > 0.0:
            adjusted_aspect = _weighted_aspect_score(item.aspect_scores, effective_weights)
            original_aspect = score_parts.get("aspect", 0.0)
            score_parts["aspect"] = row.aspect_weight * adjusted_aspect
            total_score = row.total_score - original_aspect + score_parts["aspect"]
        recommendations.append(
            DemoRecommendation(
                model_id=model_id,
                item_id=item.item_id,
                title=item.title,
                rank=row.rank,
                total_score=total_score,
                score_parts=score_parts,
                aspect_scores=dict(item.aspect_scores),
                support=row.support,
                learned_aspect_weights=dict(user.learned_aspect_weights),
                temporary_priorities=dict(validated_priorities),
                effective_aspect_weights=dict(effective_weights),
                evidence_state=evidence_state(evidence_for_item(bundle, item.item_id)),
            )
        )
    recommendations.sort(
        key=lambda recommendation: (-recommendation.total_score, recommendation.item_id)
    )
    return tuple(
        DemoRecommendation(
            model_id=recommendation.model_id,
            item_id=recommendation.item_id,
            title=recommendation.title,
            rank=index,
            total_score=recommendation.total_score,
            score_parts=recommendation.score_parts,
            aspect_scores=recommendation.aspect_scores,
            support=recommendation.support,
            learned_aspect_weights=recommendation.learned_aspect_weights,
            temporary_priorities=recommendation.temporary_priorities,
            effective_aspect_weights=recommendation.effective_aspect_weights,
            evidence_state=recommendation.evidence_state,
        )
        for index, recommendation in enumerate(recommendations[:k], start=1)
    )


def compare_rankings(
    before: Sequence[DemoRecommendation], after: Sequence[DemoRecommendation]
) -> tuple[str, ...]:
    """Return item IDs whose rank changed between two prepared views."""

    before_ranks = {recommendation.item_id: recommendation.rank for recommendation in before}
    after_ranks = {recommendation.item_id: recommendation.rank for recommendation in after}
    return tuple(
        item_id
        for item_id in sorted(set(before_ranks) & set(after_ranks))
        if before_ranks[item_id] != after_ranks[item_id]
    )


__all__ = [
    "DEFAULT_BUNDLE_PATH",
    "DemoBundle",
    "DemoEvidence",
    "DemoItem",
    "DemoLoadResult",
    "DemoMetric",
    "DemoRecommendation",
    "DemoUser",
    "DemoValidationError",
    "SUPPORT_THRESHOLD",
    "compare_rankings",
    "evidence_for_item",
    "evidence_state",
    "load_demo_bundle",
    "load_demo_bundle_with_fallback",
    "rank_recommendations",
]
