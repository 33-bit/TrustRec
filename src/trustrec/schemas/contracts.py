"""Typed contracts for snapshot-aware artifacts.

These contracts deliberately use the standard library so schema tests can run
before optional data-science dependencies are installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


def _require_non_empty(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field} must be non-empty")


@dataclass(frozen=True)
class SnapshotManifest:
    """Immutable provenance record for one time-bounded dataset snapshot."""

    snapshot_id: str
    cutoff_timestamp: datetime
    dataset_hash: str
    source_uri: str
    created_at: datetime
    row_counts: dict[str, int]

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_id, "snapshot_id")
        _require_non_empty(self.dataset_hash, "dataset_hash")
        _require_non_empty(self.source_uri, "source_uri")
        if self.cutoff_timestamp.tzinfo is None or self.created_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if self.created_at.tzinfo != UTC:
            raise ValueError("created_at must use UTC")
        if any(count < 0 for count in self.row_counts.values()):
            raise ValueError("row_counts cannot contain negative values")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cutoff_timestamp"] = self.cutoff_timestamp.isoformat()
        result["created_at"] = self.created_at.isoformat()
        return result


@dataclass(frozen=True)
class EvidenceRef:
    """A review passage that can be traced back to a recommendation claim."""

    review_id: str
    item_id: str
    aspect: str
    text: str
    sentiment: str
    confidence: float
    source_timestamp: datetime

    def __post_init__(self) -> None:
        for value, field in (
            (self.review_id, "review_id"),
            (self.item_id, "item_id"),
            (self.aspect, "aspect"),
            (self.text, "text"),
        ):
            _require_non_empty(value, field)
        if self.sentiment not in {"positive", "negative", "neutral"}:
            raise ValueError("sentiment must be positive, negative, or neutral")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.source_timestamp.tzinfo is None:
            raise ValueError("source_timestamp must be timezone-aware")


@dataclass(frozen=True)
class RecommendationRecord:
    """One ranked item and its auditable score components."""

    user_id: str
    snapshot_id: str
    item_id: str
    rank: int
    total_score: float
    component_scores: dict[str, float]
    evidence: tuple[EvidenceRef, ...]
    explanation_status: str

    def __post_init__(self) -> None:
        _require_non_empty(self.user_id, "user_id")
        _require_non_empty(self.snapshot_id, "snapshot_id")
        _require_non_empty(self.item_id, "item_id")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.explanation_status not in {"supported", "insufficient_support", "unavailable"}:
            raise ValueError("invalid explanation_status")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = [
            item.to_dict() if hasattr(item, "to_dict") else asdict(item) for item in self.evidence
        ]
        return result
