"""Assertions that enforce temporal and target/evidence isolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any


class LeakageError(AssertionError):
    """Raised when a snapshot violates a no-future-information rule."""


def assert_timestamps_before(
    rows: Iterable[Mapping[str, Any]], cutoff_timestamp_ms: int, label: str
) -> None:
    offending = [
        int(row["timestamp"])
        for row in rows
        if row.get("timestamp") is not None and int(row["timestamp"]) >= cutoff_timestamp_ms
    ]
    if offending:
        raise LeakageError(
            f"{label} contains {len(offending)} timestamps at/after cutoff "
            f"{cutoff_timestamp_ms}; maximum={max(offending)}"
        )


def assert_review_ids_disjoint(
    feature_rows: Iterable[Mapping[str, Any]], target_rows: Iterable[Mapping[str, Any]], label: str
) -> None:
    feature_ids = {row["review_id"] for row in feature_rows if row.get("review_id")}
    target_ids = {row["review_id"] for row in target_rows if row.get("review_id")}
    overlap = feature_ids & target_ids
    if overlap:
        raise LeakageError(f"{label} reuses {len(overlap)} target review IDs in features")


def parse_cutoff_ms(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return round(timestamp.timestamp() * 1000)
