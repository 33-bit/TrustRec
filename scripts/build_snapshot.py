"""Build a deterministic, bounded Parquet snapshot for a candidate category."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trustrec.data.remote_jsonl import iter_jsonl_records, sample_records, source_size

BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
EXCLUDED_ENTITY_TERMS = (
    "accessory",
    "accessories",
    "console",
    "consoles",
    "controller",
    "controllers",
    "cable",
    "cables",
    "headset",
    "headsets",
    "headphone",
    "shipping",
    "packaging",
    "seller",
    "sellers",
    "keyboard",
    "mouse",
    "charger",
    "adapter",
    "transmitter",
    "remote",
    "stand",
    "case",
    "book",
    "guide",
    "magazine",
    "coloring",
    "workbook",
    "battery",
    "sleeve",
    "protector",
    "dock",
)
ALLOWED_ITEM_TYPES = {
    "video game",
    "software download",
    "game",
    "computer game",
    "cd-rom",
    "dvd-rom",
    "game cartridge",
    "software",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def stable_review_id(category: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({"category": category, "row": row})).hexdigest()[:24]


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def text_duplicate_group(value: str | None, item_id: str | None = None) -> str:
    """Return an item-scoped group for repeated normalized review text."""

    payload = {"item_id": item_id or "", "text": normalize_text(value).lower()}
    return hashlib.sha256(canonical_json(payload)).hexdigest()[:24]


def is_software_game(
    metadata: dict[str, Any] | None,
    excluded_terms: tuple[str, ...] = EXCLUDED_ENTITY_TERMS,
) -> bool:
    """Keep software games and reject hardware, books, and merchant records."""

    if not metadata:
        return False
    title = str(metadata.get("title") or "").casefold()
    category = str(metadata.get("main_category") or "").casefold()
    categories = str(metadata.get("categories") or "").casefold()
    details = parse_json_mapping(metadata.get("details") or metadata.get("details_json"))
    features = parse_json_list(metadata.get("features") or metadata.get("features_json"))
    title_category_text = " ".join((title, category, categories)).casefold()
    if any(re.search(rf"\b{re.escape(term)}\b", title_category_text) for term in excluded_terms):
        return False
    item_type = ""
    for key, value in details.items():
        if str(key).casefold() == "type of item":
            item_type = str(value or "").casefold().strip()
            break
    if item_type:
        if item_type in ALLOWED_ITEM_TYPES:
            return True
        return False
    scope_text = " ".join((title, category, categories, " ".join(map(str, features)))).casefold()
    if any(re.search(rf"\b{re.escape(term)}\b", scope_text) for term in excluded_terms):
        return False
    return category == "video games" and bool(scope_text.strip())


def validate_review_row(category: str, row: dict[str, Any]) -> None:
    """Reject records that cannot participate in a time-safe snapshot."""

    review_id = stable_review_id(category, row)
    required = {
        "user_id": row.get("user_id"),
        "item_id": row.get("parent_asin") or row.get("asin") or row.get("item_id"),
        "timestamp": row.get("timestamp"),
        "rating": row.get("rating"),
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(f"review {review_id} is missing required fields: {', '.join(missing)}")


def quantile(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("at least one timestamp is required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def json_value(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_json_mapping(value: Any) -> dict[str, Any]:
    """Return a metadata mapping from a dictionary or serialized JSON value."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_json_list(value: Any) -> list[Any]:
    """Return a metadata list from a list or serialized JSON value."""

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", version="2.6")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_interactions(
    interactions: list[dict[str, Any]], cutoff_t0: int, cutoff_t1: int
) -> dict[str, list[dict[str, Any]]]:
    """Create time-safe tables and keep the first target per user-item pair."""

    if cutoff_t0 >= cutoff_t1:
        raise ValueError("cutoff_t0 must be earlier than cutoff_t1")
    for row in interactions:
        missing = [
            field
            for field in ("review_id", "user_id", "item_id", "timestamp")
            if row.get(field) in (None, "")
        ]
        if missing:
            raise ValueError("interaction is missing required fields: " + ", ".join(missing))
    ordered = sorted(
        interactions,
        key=lambda row: (
            int(row["timestamp"]),
            str(row["user_id"]),
            str(row["item_id"]),
            str(row["review_id"]),
        ),
    )

    train = [row for row in ordered if row["timestamp"] < cutoff_t0]
    fit = [row for row in ordered if row["timestamp"] < cutoff_t1]
    first_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ordered:
        first_by_pair.setdefault((row["user_id"], row["item_id"]), row)

    def target_rows(
        history_rows: list[dict[str, Any]], lower: int, upper: int | None
    ) -> list[dict[str, Any]]:
        history = {}
        catalog = set()
        for row in history_rows:
            history.setdefault(row["user_id"], set()).add(row["item_id"])
            catalog.add(row["item_id"])
        selected = []
        seen_pairs = set()
        for row in first_by_pair.values():
            timestamp = row["timestamp"]
            in_window = timestamp >= lower and (upper is None or timestamp < upper)
            pair = (row["user_id"], row["item_id"])
            if (
                in_window
                and row["rating"] is not None
                and row["rating"] >= 4
                and row["item_id"] in catalog
                and row["item_id"] not in history.get(row["user_id"], set())
                and pair not in seen_pairs
            ):
                selected.append(row)
                seen_pairs.add(pair)
        return selected

    return {
        "train_interactions": train,
        "fit_interactions": fit,
        "validation_targets": target_rows(train, cutoff_t0, cutoff_t1),
        "test_targets": target_rows(fit, cutoff_t1, None),
    }


def build_snapshot(
    category: str,
    output_root: Path,
    manifest_dir: Path,
    window_bytes: int,
    max_reviews: int,
    metadata_mode: str,
    excluded_terms: tuple[str, ...] = EXCLUDED_ENTITY_TERMS,
) -> dict[str, Any]:
    review_url = f"{BASE_URL}/raw/review_categories/{category}.jsonl"
    metadata_url = f"{BASE_URL}/raw/meta_categories/meta_{category}.jsonl"
    review_rows, review_source = sample_records(review_url, window_bytes=window_bytes)
    if metadata_mode == "full":
        metadata_rows = []
        # Review item IDs are known only after the bounded review sample is read.
        metadata_source = {
            "source_url": metadata_url,
            "source_bytes": source_size(metadata_url),
            "scan": "full_sequential_range_scan",
            "chunk_bytes": window_bytes,
        }
        _, metadata_windows = sample_records(metadata_url, window_bytes=window_bytes)
        metadata_source.update(
            {
                "window_hashes": metadata_windows["window_hashes"],
                "window_digest": metadata_windows["window_digest"],
                "window_starts": metadata_windows["window_starts"],
                "window_count": metadata_windows["window_count"],
            }
        )
    else:
        metadata_rows, metadata_source = sample_records(metadata_url, window_bytes=window_bytes)

    for row in review_rows:
        validate_review_row(category, row)
    reviews_by_id = {stable_review_id(category, row): row for row in review_rows}

    metadata_item_ids = {
        row.get("parent_asin") or row.get("asin")
        for row in review_rows
        if row.get("parent_asin") or row.get("asin")
    }
    if metadata_mode == "full":
        metadata_rows = [
            row
            for row in iter_jsonl_records(metadata_url, chunk_bytes=window_bytes)
            if (row.get("parent_asin") or row.get("asin")) in metadata_item_ids
        ]
        metadata_source["matched_rows_scanned"] = len(metadata_rows)

    metadata_by_item = {
        row.get("parent_asin") or row.get("asin"): row
        for row in metadata_rows
        if row.get("parent_asin") or row.get("asin")
    }
    scoped_review_rows = [
        row
        for row in reviews_by_id.values()
        if is_software_game(
            metadata_by_item.get(row.get("parent_asin") or row.get("asin")), excluded_terms
        )
    ]
    review_rows = sorted(
        scoped_review_rows,
        key=lambda row: (
            int(row["timestamp"]),
            str(row["user_id"]),
            str(row.get("parent_asin") or row.get("asin")),
            stable_review_id(category, row),
        ),
    )[:max_reviews]
    review_ids = {stable_review_id(category, row): row for row in review_rows}
    interactions = []
    review_texts = []
    for review_id, row in sorted(
        review_ids.items(),
        key=lambda pair: (
            int(pair[1]["timestamp"]),
            str(pair[1]["user_id"]),
            str(pair[1].get("parent_asin") or pair[1].get("asin")),
            pair[0],
        ),
    ):
        item_id = row.get("parent_asin") or row.get("asin")
        text = row.get("text")
        interactions.append(
            {
                "review_id": review_id,
                "user_id": row.get("user_id"),
                "item_id": item_id,
                "rating": float(row["rating"]),
                "timestamp": int(row["timestamp"]),
            }
        )
        review_texts.append(
            {
                "review_id": review_id,
                "item_id": item_id,
                "title": row.get("title"),
                "text": text,
                "normalized_text": normalize_text(text),
                "language": None,
                "timestamp": int(row["timestamp"]),
                "source_text_sha256": hashlib.sha256((text or "").encode()).hexdigest(),
                "duplicate_group": text_duplicate_group(text, item_id),
            }
        )

    item_ids = sorted({row["item_id"] for row in interactions if row["item_id"]})
    item_metadata = []
    for item_id in item_ids:
        row = metadata_by_item.get(item_id, {})
        item_metadata.append(
            {
                "item_id": item_id,
                "parent_asin": item_id,
                "main_category": scalar_text(row.get("main_category")),
                "title": scalar_text(row.get("title")),
                "description_json": json_value(row.get("description")),
                "features_json": json_value(row.get("features")),
                "price": scalar_text(row.get("price")),
                "brand": scalar_text(row.get("brand")),
                "images_json": json_value(row.get("images")),
                "details_json": json_value(row.get("details")),
                "metadata_found": bool(row),
                "entity_scope": "software_game"
                if is_software_game(row, excluded_terms)
                else "out_of_scope",
            }
        )

    timestamps = [row["timestamp"] for row in interactions]
    cutoff_t0 = quantile(timestamps, 0.80)
    cutoff_t1 = quantile(timestamps, 0.90)
    if cutoff_t0 >= cutoff_t1:
        raise ValueError("the source does not provide distinct T0 and T1 cutoffs")
    split_rows = split_interactions(interactions, cutoff_t0, cutoff_t1)
    canonical_rows = {
        "interactions": interactions,
        "review_texts": review_texts,
        "item_metadata": item_metadata,
        **split_rows,
    }
    dataset_hash = hashlib.sha256(canonical_json(canonical_rows)).hexdigest()
    snapshot_id = f"{category.lower()}-pilot-{dataset_hash[:12]}"
    snapshot_dir = output_root / category.lower() / snapshot_id
    artifact_paths = {name: snapshot_dir / f"{name}.parquet" for name in canonical_rows}
    artifact_hashes = {
        name: write_parquet(path, canonical_rows[name]) for name, path in artifact_paths.items()
    }
    metadata_found = sum(1 for row in item_metadata if row["metadata_found"])
    created_at = datetime.now(UTC)
    manifest = {
        "schema_version": 2,
        "snapshot_id": snapshot_id,
        "category": category,
        "artifact_role": "development_pilot",
        "benchmark_eligible": False,
        "entity_scope_status": "software_games_only_after_metadata_join",
        "metadata_usage": "scope_and_display_only",
        "created_at": created_at.isoformat(),
        "profile_type": "three-byte-window-pilot",
        "cutoffs": {
            "t0": datetime.fromtimestamp(cutoff_t0 / 1000, UTC).isoformat(),
            "t1": datetime.fromtimestamp(cutoff_t1 / 1000, UTC).isoformat(),
            "policy": "global_timestamp_quantiles_0.80_0.90",
        },
        "dataset_hash": dataset_hash,
        "row_counts": {name: len(rows) for name, rows in canonical_rows.items()},
        "unique_counts": {
            "users": len({row["user_id"] for row in interactions if row["user_id"]}),
            "items": len(item_ids),
            "metadata_matched_items": metadata_found,
            "validation_target_users": len(
                {row["user_id"] for row in split_rows["validation_targets"]}
            ),
            "test_target_users": len({row["user_id"] for row in split_rows["test_targets"]}),
        },
        "metadata_coverage": round(metadata_found / len(item_ids), 6) if item_ids else 0.0,
        "sources": {"reviews": review_source, "metadata": metadata_source},
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_sha256": artifact_hashes,
        "sampling": {
            "window_bytes": window_bytes,
            "max_reviews": max_reviews,
            "selection": "deduplicate_by_stable_review_id_then_sort_by_timestamp_user_item_review",
        },
        "scope_counts": {
            "sampled_reviews": len(reviews_by_id),
            "software_game_reviews": len(review_rows),
            "excluded_reviews": len(reviews_by_id) - len(review_rows),
        },
    }
    manifest_path = manifest_dir / f"{snapshot_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--window-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-reviews", type=int, default=200_000)
    parser.add_argument("--metadata-mode", choices=("full", "windows"), default="full")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    manifest = build_snapshot(
        category=args.category,
        output_root=args.output_root,
        manifest_dir=args.manifest_dir,
        window_bytes=args.window_bytes,
        max_reviews=args.max_reviews,
        metadata_mode=args.metadata_mode,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
