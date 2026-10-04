"""Assertions that enforce temporal and evaluation-set isolation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any


class LeakageError(AssertionError):
    """Raised when a snapshot violates a no-future-information rule."""


def _required_review_id(row: Mapping[str, Any], label: str) -> str:
    review_id = row.get("review_id")
    if review_id in (None, ""):
        raise LeakageError(f"{label} has a row without review_id")
    return str(review_id)


def _required_timestamp(row: Mapping[str, Any], label: str) -> int:
    value = row.get("timestamp", row.get("source_timestamp"))
    if value in (None, ""):
        raise LeakageError(f"{label} has a row without timestamp")
    try:
        if isinstance(value, datetime):
            return round(value.timestamp() * 1000)
        if isinstance(value, str) and "T" in value:
            return parse_cutoff_ms(value)
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise LeakageError(f"{label} has a non-numeric timestamp") from error


def assert_timestamps_before(
    rows: Iterable[Mapping[str, Any]],
    cutoff_timestamp_ms: int,
    label: str,
    *,
    require_review_id: bool = True,
) -> None:
    offending: list[int] = []
    for row in rows:
        if require_review_id:
            _required_review_id(row, label)
        timestamp = _required_timestamp(row, label)
        if timestamp >= cutoff_timestamp_ms:
            offending.append(timestamp)
    if offending:
        raise LeakageError(
            f"{label} contains {len(offending)} timestamps at/after cutoff "
            f"{cutoff_timestamp_ms}; maximum={max(offending)}"
        )


def assert_review_ids_disjoint(
    feature_rows: Iterable[Mapping[str, Any]], target_rows: Iterable[Mapping[str, Any]], label: str
) -> None:
    feature_ids = {_required_review_id(row, f"{label} features") for row in feature_rows}
    target_ids = {_required_review_id(row, f"{label} targets") for row in target_rows}
    overlap = feature_ids & target_ids
    if overlap:
        raise LeakageError(f"{label} reuses {len(overlap)} target review IDs in features")


def assert_metadata_before(
    metadata_rows: Iterable[Mapping[str, Any]], cutoff_timestamp_ms: int, label: str = "metadata"
) -> None:
    """Reject metadata without a timestamp or with a post-cutoff timestamp."""

    for row in metadata_rows:
        if row.get("item_id") in (None, ""):
            raise LeakageError(f"{label} has a row without item_id")
        value = row.get("metadata_timestamp", row.get("timestamp"))
        if value in (None, ""):
            raise LeakageError(f"{label} has a row without metadata_timestamp")
        try:
            timestamp = parse_cutoff_ms(value) if isinstance(value, str) else int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise LeakageError(f"{label} has a non-numeric metadata_timestamp") from error
        if timestamp >= cutoff_timestamp_ms:
            raise LeakageError(f"{label} contains post-cutoff metadata")


def _set_from_rows(
    rows: Iterable[Mapping[str, Any]] | Iterable[Any], field: str, label: str
) -> set[Any]:
    values = set()
    for row in rows:
        if isinstance(row, Mapping):
            if field not in row:
                raise LeakageError(f"{label} has a row without {field}")
            value = row[field]
        else:
            value = row
        try:
            values.add(value if not isinstance(value, list) else tuple(value))
        except TypeError as error:
            raise LeakageError(f"{label} contains an unhashable {field}") from error
    return values


def assert_same_evaluation_sets(
    model_results: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
) -> None:
    """Require every compared model to use the same users, candidates, and targets."""

    if not model_results:
        raise LeakageError("no model evaluation sets were supplied")
    fields = (
        ("user_id", "users"),
        ("candidate_set", "candidates"),
        ("target_set", "targets"),
    )
    baseline_name, baseline = next(iter(model_results.items()))
    for field, alias in fields:
        baseline_rows = baseline.get(field, baseline.get(alias))
        if baseline_rows is None:
            raise LeakageError(f"{baseline_name} lacks {field}")
        baseline_values = _set_from_rows(baseline_rows, field, baseline_name)
        for name, result in model_results.items():
            rows = result.get(field, result.get(alias))
            if rows is None:
                raise LeakageError(f"{name} lacks {field}")
            values = _set_from_rows(rows, field, name)
            if values != baseline_values:
                raise LeakageError(f"models use different {field} sets: {baseline_name} and {name}")


def assert_pseudo_test_not_used_for_tuning(manifest: Mapping[str, Any]) -> None:
    """Reject a frozen LLM pseudo-test manifest that allows tuning."""

    if manifest.get("split_role") != "llm_pseudo_test":
        return
    restrictions = manifest.get("usage_restrictions", {})
    allowed = restrictions.get("allowed_uses", []) if isinstance(restrictions, Mapping) else []
    if manifest.get("frozen", False) and any(
        term in " ".join(map(str, allowed)).casefold()
        for term in ("tuning", "prompt", "threshold", "hyperparameter")
    ):
        raise LeakageError("llm_pseudo_test cannot be used for tuning")
    if manifest.get("tuning_allowed") is True or manifest.get("prompt_selection_allowed") is True:
        raise LeakageError("llm_pseudo_test cannot be used for tuning")
    for key in (
        "tuning",
        "prompt_tuning",
        "threshold_tuning",
        "model_tuning",
        "hyperparameter_tuning",
    ):
        if manifest.get(key):
            raise LeakageError("llm_pseudo_test cannot be used for tuning")


def parse_cutoff_ms(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("cutoff must include a time zone")
    return round(timestamp.timestamp() * 1000)
