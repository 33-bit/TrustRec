"""Shared contracts for snapshot-aware ranking baselines."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

Timestamp = int | float | str | datetime


def timestamp_value(value: Timestamp) -> float:
    """Convert supported timestamps to epoch milliseconds for comparisons."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.timestamp() * 1000.0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp must be non-empty")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                numeric = float(text)
            except ValueError as error:
                raise ValueError("timestamp must be numeric or ISO formatted") from error
            if not math.isfinite(numeric):
                raise ValueError("timestamp must be finite") from None
            return numeric
        if parsed.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return parsed.timestamp() * 1000.0
    if isinstance(value, bool):
        raise ValueError("timestamp must be numeric or ISO formatted")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp must be numeric or ISO formatted") from error
    if not math.isfinite(numeric):
        raise ValueError("timestamp must be finite")
    return numeric


def _require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def coerce_interactions(rows: Iterable[Mapping[str, Any]] | Any) -> tuple[dict[str, Any], ...]:
    """Normalize mapping rows from lists or pandas-like objects."""

    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    normalized: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            try:
                row = dict(row)
            except (TypeError, ValueError) as error:
                raise TypeError(f"interaction row {position} must be a mapping") from error
        copied = dict(row)
        for field in ("user_id", "item_id", "timestamp"):
            if field not in copied or copied[field] in (None, ""):
                raise ValueError(f"interaction row {position} is missing {field}")
        _require_id(copied["user_id"], "user_id")
        _require_id(copied["item_id"], "item_id")
        timestamp_value(copied["timestamp"])
        normalized.append(copied)
    return tuple(normalized)


def first_interactions_before_cutoff(
    rows: Iterable[Mapping[str, Any]] | Any,
    cutoff_timestamp: Timestamp,
) -> tuple[dict[str, Any], ...]:
    """Keep the first time-ordered event for each user-item pair."""

    cutoff_value = timestamp_value(cutoff_timestamp)
    first_by_pair: dict[tuple[str, str], tuple[tuple[float, str, str], dict[str, Any]]] = {}
    for row in coerce_interactions(rows):
        row_timestamp = timestamp_value(row["timestamp"])
        if row_timestamp >= cutoff_value:
            continue
        key = (row["user_id"], row["item_id"])
        canonical_row = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        order = (row_timestamp, str(row.get("review_id", "")), canonical_row)
        if key not in first_by_pair or order < first_by_pair[key][0]:
            first_by_pair[key] = (order, row)
    return tuple(
        row
        for _, row in sorted(
            first_by_pair.values(),
            key=lambda value: (value[0], value[1]["user_id"], value[1]["item_id"]),
        )
    )


@dataclass(frozen=True)
class CandidateSet:
    """The user and exact candidate IDs shared by compared rankers."""

    user_id: str
    snapshot_id: str
    cutoff_timestamp: Timestamp
    candidate_item_ids: tuple[str, ...]
    history_item_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _require_id(self.user_id, "user_id")
        _require_id(self.snapshot_id, "snapshot_id")
        timestamp_value(self.cutoff_timestamp)
        candidate_ids = tuple(self.candidate_item_ids)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_item_ids must be unique")
        for item_id in candidate_ids:
            _require_id(item_id, "candidate item_id")
        history = frozenset(self.history_item_ids)
        for item_id in history:
            _require_id(item_id, "history item_id")
        if set(candidate_ids) & history:
            raise ValueError("candidate_item_ids must exclude history_item_ids")
        object.__setattr__(self, "candidate_item_ids", candidate_ids)
        object.__setattr__(self, "history_item_ids", history)


@dataclass(frozen=True)
class RankedItem:
    """One ranked candidate and its auditable component scores."""

    item_id: str
    rank: int
    total_score: float
    component_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_id(self.item_id, "item_id")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if not math.isfinite(float(self.total_score)):
            raise ValueError("total_score must be finite")
        scores = {str(name): float(value) for name, value in self.component_scores.items()}
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("component scores must be finite")
        object.__setattr__(self, "total_score", float(self.total_score))
        object.__setattr__(self, "component_scores", scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "rank": self.rank,
            "total_score": self.total_score,
            "component_scores": dict(self.component_scores),
        }


@dataclass(frozen=True)
class RankingResult:
    """Auditable output from one baseline for one candidate set."""

    model_id: str
    user_id: str
    snapshot_id: str
    cutoff_timestamp: Timestamp
    candidate_item_ids: tuple[str, ...]
    items: tuple[RankedItem, ...]
    configuration: Mapping[str, Any]
    seed: int | None
    model_hash: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.model_id, "model_id")
        _require_id(self.user_id, "user_id")
        _require_id(self.snapshot_id, "snapshot_id")
        timestamp_value(self.cutoff_timestamp)
        candidate_ids = tuple(self.candidate_item_ids)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_item_ids must be unique")
        for item_id in candidate_ids:
            _require_id(item_id, "candidate item_id")
        items = tuple(self.items)
        if any(item.item_id not in candidate_ids for item in items):
            raise ValueError("ranked items must belong to candidate_item_ids")
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("ranked items must be unique")
        if tuple(item.rank for item in items) != tuple(range(1, len(items) + 1)):
            raise ValueError("ranked item ranks must be contiguous and one-based")
        _require_id(self.model_hash, "model_hash")
        object.__setattr__(self, "candidate_item_ids", candidate_ids)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "configuration", dict(self.configuration))

    def to_dict(self) -> dict[str, Any]:
        cutoff = self.cutoff_timestamp
        if isinstance(cutoff, datetime):
            cutoff = cutoff.isoformat()
        return {
            "model_id": self.model_id,
            "user_id": self.user_id,
            "snapshot_id": self.snapshot_id,
            "cutoff_timestamp": cutoff,
            "candidate_item_ids": list(self.candidate_item_ids),
            "items": [item.to_dict() for item in self.items],
            "configuration": dict(self.configuration),
            "seed": self.seed,
            "model_hash": self.model_hash,
            "fallback_reason": self.fallback_reason,
        }


def build_candidate_set(
    user_id: str,
    snapshot_id: str,
    interactions: Iterable[Mapping[str, Any]] | Any,
    cutoff_timestamp: Timestamp,
    candidate_item_ids: Iterable[str] | None = None,
) -> CandidateSet:
    """Build known-item candidates by removing the user's pre-cutoff history."""

    before_cutoff = first_interactions_before_cutoff(interactions, cutoff_timestamp)
    catalog = {row["item_id"] for row in before_cutoff}
    history = {row["item_id"] for row in before_cutoff if row["user_id"] == user_id}
    if candidate_item_ids is None:
        candidates = sorted(catalog - history)
    else:
        candidates = list(candidate_item_ids)
        unknown = set(candidates) - catalog
        if unknown:
            raise ValueError("candidate_item_ids must be known before the cutoff")
    return CandidateSet(
        user_id=user_id,
        snapshot_id=snapshot_id,
        cutoff_timestamp=cutoff_timestamp,
        candidate_item_ids=tuple(candidates),
        history_item_ids=frozenset(history),
    )


def stable_model_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible model payload with stable key ordering."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
