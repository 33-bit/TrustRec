"""Validation and loading for development and pseudo-test label records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

ASPECTS = (
    "gameplay",
    "story",
    "graphics",
    "performance",
    "controls",
    "multiplayer",
    "content_replay",
    "value",
)
SENTIMENTS = ("positive", "negative", "neutral")


def parse_cutoff_timestamp(value: str) -> int:
    """Convert a timezone-aware ISO-8601 timestamp to milliseconds."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("source_cutoff_timestamp must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("source_cutoff_timestamp must include a time zone")
    return round(parsed.timestamp() * 1000)


def source_hash(text: str) -> str:
    """Return the SHA-256 hash required by the annotation contract."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records from ``path``."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(row)
    return rows


def _validate_label(label: dict[str, Any], text: str, source: str) -> None:
    required = {"aspect", "polarity", "start", "end", "evidence"}
    missing = sorted(required - set(label))
    if missing:
        raise ValueError(f"{source} label lacks required fields: {', '.join(missing)}")
    if label["aspect"] not in ASPECTS:
        raise ValueError(f"{source} has unsupported aspect {label['aspect']!r}")
    if label["polarity"] not in SENTIMENTS:
        raise ValueError(f"{source} has unsupported polarity {label['polarity']!r}")
    if not isinstance(label["start"], int) or not isinstance(label["end"], int):
        raise ValueError(f"{source} evidence offsets must be integers")
    start, end = label["start"], label["end"]
    if start < 0 or end < start or end > len(text):
        raise ValueError(f"{source} evidence offsets are outside source text")
    if text[start:end] != label["evidence"]:
        raise ValueError(f"{source} evidence does not match source text")


def validate_annotation_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_role: str,
    expected_status: str,
    reject_cutoff: bool = True,
) -> list[dict[str, Any]]:
    """Validate label records and return them in input order."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one annotation record is required")
    seen_units: set[str] = set()
    for index, record in enumerate(rows, start=1):
        source = f"record {index}"
        required = {
            "unit_id",
            "review_id",
            "item_id",
            "text",
            "unit_text",
            "unit_start",
            "unit_end",
            "labels",
            "split_role",
            "label_status",
            "timestamp",
            "source_cutoff_timestamp",
            "source_text_sha256",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"{source} lacks required fields: {', '.join(missing)}")
        unit_id = record["unit_id"]
        if unit_id in seen_units:
            raise ValueError(f"duplicate unit_id: {unit_id}")
        seen_units.add(unit_id)
        if record["split_role"] != expected_role:
            raise ValueError(f"{source} must use split role {expected_role}")
        if record["label_status"] != expected_status:
            raise ValueError(f"{source} must use label status {expected_status}")
        text = record["text"]
        if not isinstance(text, str) or not text:
            raise ValueError(f"{source} text must be non-empty")
        if source_hash(text) != record["source_text_sha256"]:
            raise ValueError(f"{source} source_text_sha256 does not match text")
        start, end = record["unit_start"], record["unit_end"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end > len(text)
        ):
            raise ValueError(f"{source} unit offsets are invalid")
        if text[start:end] != record["unit_text"]:
            raise ValueError(f"{source} unit_text does not match text")
        cutoff = parse_cutoff_timestamp(record["source_cutoff_timestamp"])
        if reject_cutoff and (
            not isinstance(record["timestamp"], int) or record["timestamp"] >= cutoff
        ):
            raise ValueError(f"{source} timestamp must be before the source cutoff")
        labels = record["labels"]
        if not isinstance(labels, list):
            raise ValueError(f"{source} labels must be a list")
        for label in labels:
            if not isinstance(label, dict):
                raise ValueError(f"{source} labels must contain objects")
            _validate_label(label, text, source)
            if not start <= label["start"] <= label["end"] <= end:
                raise ValueError(f"{source} evidence must be inside unit_text")
    return rows


def assert_disjoint_splits(
    training_records: Iterable[dict[str, Any]],
    evaluation_records: Iterable[dict[str, Any]],
) -> None:
    """Reject review and item-scoped duplicate overlap between splits."""

    train = list(training_records)
    evaluation = list(evaluation_records)
    train_reviews = {row["review_id"] for row in train}
    eval_reviews = {row["review_id"] for row in evaluation}
    overlap = sorted(train_reviews & eval_reviews)
    if overlap:
        raise ValueError("training and evaluation review overlap: " + ", ".join(overlap[:3]))
    train_groups = {
        (row.get("item_id"), row.get("duplicate_group"))
        for row in train
        if row.get("duplicate_group") not in (None, "")
    }
    eval_groups = {
        (row.get("item_id"), row.get("duplicate_group"))
        for row in evaluation
        if row.get("duplicate_group") not in (None, "")
    }
    if overlap_groups := sorted(train_groups & eval_groups):
        raise ValueError(f"training and evaluation duplicate-group overlap: {overlap_groups[0]}")


def training_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep usable development records for optional NLP training."""

    rows = [row for row in records if not row.get("out_of_scope") and not row.get("needs_review")]
    if not any(row.get("labels") for row in rows):
        raise ValueError("development_pilot has no usable labels")
    return rows
