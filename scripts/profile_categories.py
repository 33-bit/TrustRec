"""Profile Amazon Reviews 2023 candidate categories without downloading full files.

The script reads three deterministic byte windows from the public JSONL review
and metadata files. Results are explicitly labelled as a pilot profile: unique
counts and retention are lower-bound/sample estimates, while the source card's
full-category counts are recorded separately in the decision memo.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustrec.data.remote_jsonl import sample_records

BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
WINDOW_BYTES = 16 * 1024 * 1024
WINDOW_FRACTIONS = (0.0, 0.5, 1.0)


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


def quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def timestamp_ms(row: dict[str, Any]) -> int | None:
    value = row.get("timestamp")
    return int(value) if value is not None else None


def retention_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = [
        (
            row.get("user_id"),
            row.get("parent_asin") or row.get("asin"),
            float(row.get("rating", 0)),
            timestamp_ms(row),
        )
        for row in rows
    ]
    events = [event for event in events if all(value is not None for value in event)]
    timestamps = [event[3] for event in events]
    if len(timestamps) < 10:
        return {"status": "insufficient_timestamped_rows", "rows": len(events)}

    t0 = quantile(timestamps, 0.80)
    t1 = max(t0 + 1, quantile(timestamps, 0.90))

    def window_stats(cutoff: int, target_end: int | None) -> dict[str, Any]:
        history = defaultdict(set)
        catalog: set[str] = set()
        for user, item, _, timestamp in events:
            if timestamp < cutoff:
                history[user].add(item)
                catalog.add(item)

        users_with_history = set(history)
        target_pairs: set[tuple[str, str]] = set()
        for user, item, rating, timestamp in events:
            in_window = timestamp >= cutoff and (target_end is None or timestamp < target_end)
            if in_window and rating >= 4 and item in catalog and item not in history[user]:
                target_pairs.add((user, item))
        eligible_users = {user for user, _ in target_pairs}
        history_groups = Counter(
            "1-2" if len(items) <= 2 else "3-5" if len(items) <= 5 else ">5"
            for items in history.values()
        )
        return {
            "users_with_history": len(users_with_history),
            "eligible_users": len(eligible_users),
            "eligible_fraction_of_history_users": round(
                len(eligible_users) / len(users_with_history), 6
            )
            if users_with_history
            else 0.0,
            "unique_target_user_item_pairs": len(target_pairs),
            "known_items_at_cutoff": len(catalog),
            "history_groups": dict(history_groups),
        }

    return {
        "cutoff_t0": datetime.fromtimestamp(t0 / 1000, UTC).isoformat(),
        "cutoff_t1": datetime.fromtimestamp(t1 / 1000, UTC).isoformat(),
        "validation": window_stats(t0, t1),
        "test": window_stats(t1, None),
    }


def profile_reviews(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = ("rating", "title", "text", "parent_asin", "user_id", "timestamp")
    missing = Counter()
    ratings = Counter()
    users: set[str] = set()
    items: set[str] = set()
    timestamps: list[int] = []
    text_lengths: list[int] = []
    for row in rows:
        for field in field_names:
            if is_missing(row.get(field)):
                missing[field] += 1
        if row.get("rating") is not None:
            ratings[str(row["rating"])] += 1
        if row.get("user_id"):
            users.add(row["user_id"])
        if row.get("parent_asin") or row.get("asin"):
            items.add(row.get("parent_asin") or row.get("asin"))
        timestamp = timestamp_ms(row)
        if timestamp is not None:
            timestamps.append(timestamp)
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            text_lengths.append(len(text))

    return {
        "rows_sampled": len(rows),
        "unique_users_sampled": len(users),
        "unique_items_sampled": len(items),
        "missingness": {
            field: round(missing[field] / len(rows), 6) if rows else 0.0 for field in field_names
        },
        "rating_distribution": dict(sorted(ratings.items(), key=lambda pair: float(pair[0]))),
        "timestamp_range": {
            "min": datetime.fromtimestamp(min(timestamps) / 1000, UTC).isoformat()
            if timestamps
            else None,
            "max": datetime.fromtimestamp(max(timestamps) / 1000, UTC).isoformat()
            if timestamps
            else None,
        },
        "review_text_chars": {
            "non_empty_rows": len(text_lengths),
            "mean": round(statistics.mean(text_lengths), 2) if text_lengths else 0.0,
            "median": round(statistics.median(text_lengths), 2) if text_lengths else 0.0,
        },
        "retention": retention_profile(rows),
    }


def profile_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "parent_asin",
        "main_category",
        "title",
        "description",
        "features",
        "price",
        "brand",
        "images",
        "details",
    )
    missing = Counter()
    categories = Counter()
    items: set[str] = set()
    for row in rows:
        for field in fields:
            if is_missing(row.get(field)):
                missing[field] += 1
        if row.get("parent_asin"):
            items.add(row["parent_asin"])
        if row.get("main_category"):
            categories[row["main_category"]] += 1
    return {
        "rows_sampled": len(rows),
        "unique_items_sampled": len(items),
        "missingness": {
            field: round(missing[field] / len(rows), 6) if rows else 0.0 for field in fields
        },
        "main_category_top": categories.most_common(10),
    }


def category_profile(category: str) -> dict[str, Any]:
    review_url = f"{BASE_URL}/raw/review_categories/{category}.jsonl"
    meta_url = f"{BASE_URL}/raw/meta_categories/meta_{category}.jsonl"
    review_rows, review_source = sample_records(
        review_url, window_bytes=WINDOW_BYTES, window_fractions=WINDOW_FRACTIONS
    )
    meta_rows, meta_source = sample_records(
        meta_url, window_bytes=WINDOW_BYTES, window_fractions=WINDOW_FRACTIONS
    )
    return {
        "category": category,
        "profile_type": "three-byte-window-pilot",
        "review_source": review_source,
        "metadata_source": meta_source,
        "reviews": profile_reviews(review_rows),
        "metadata": profile_metadata(meta_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", dest="categories", default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/tasks/t1_1_category_profile.json")
    )
    args = parser.parse_args()
    categories = args.categories or ["Video_Games", "Electronics"]
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_bytes": WINDOW_BYTES,
        "categories": [category_profile(category) for category in categories],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
