"""Build a full, time-safe Video Games snapshot with bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scripts.build_snapshot import (
        BASE_URL,
        EXCLUDED_ENTITY_TERMS,
        canonical_json,
        is_software_game,
        json_value,
        normalize_text,
        stable_review_id,
        text_duplicate_group,
        validate_review_row,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_snapshot import (
        BASE_URL,
        EXCLUDED_ENTITY_TERMS,
        canonical_json,
        is_software_game,
        json_value,
        normalize_text,
        stable_review_id,
        text_duplicate_group,
        validate_review_row,
    )
from trustrec.data.remote_jsonl import iter_jsonl_records

INTERACTION_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("rating", pa.float64()),
        ("timestamp", pa.int64()),
    ]
)
REVIEW_TEXT_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("item_id", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("normalized_text", pa.string()),
        ("language", pa.string()),
        ("timestamp", pa.int64()),
        ("source_text_sha256", pa.string()),
        ("duplicate_group", pa.string()),
    ]
)
METADATA_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("parent_asin", pa.string()),
        ("main_category", pa.string()),
        ("title", pa.string()),
        ("description_json", pa.string()),
        ("features_json", pa.string()),
        ("price", pa.string()),
        ("brand", pa.string()),
        ("images_json", pa.string()),
        ("details_json", pa.string()),
        ("metadata_found", pa.bool_()),
        ("entity_scope", pa.string()),
    ]
)


def write_batches(
    path: Path,
    schema: pa.Schema,
    rows: Iterable[dict[str, Any]],
    batch_size: int,
) -> int:
    """Write rows to Parquet without retaining the full table in memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, schema, compression="zstd", version="2.6")
    batch: list[dict[str, Any]] = []
    count = 0
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            count += len(batch)
    finally:
        writer.close()
    return count


def metadata_row(item_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "parent_asin": item_id,
        "main_category": str(row.get("main_category")) if row.get("main_category") else None,
        "title": str(row.get("title")) if row.get("title") else None,
        "description_json": json_value(row.get("description")),
        "features_json": json_value(row.get("features")),
        "price": str(row.get("price")) if row.get("price") is not None else None,
        "brand": str(row.get("brand")) if row.get("brand") else None,
        "images_json": json_value(row.get("images")),
        "details_json": json_value(row.get("details")),
        "metadata_found": True,
        "entity_scope": "software_game",
    }


def interaction_row(category: str, row: dict[str, Any]) -> dict[str, Any]:
    validate_review_row(category, row)
    item_id = row.get("parent_asin") or row.get("asin") or row.get("item_id")
    return {
        "review_id": stable_review_id(category, row),
        "user_id": str(row["user_id"]),
        "item_id": str(item_id),
        "rating": float(row["rating"]),
        "timestamp": int(row["timestamp"]),
    }


def review_text_row(category: str, row: dict[str, Any]) -> dict[str, Any]:
    item_id = str(row.get("parent_asin") or row.get("asin") or row.get("item_id"))
    review_id = stable_review_id(category, row)
    text = row.get("text")
    return {
        "review_id": review_id,
        "item_id": item_id,
        "title": str(row.get("title")) if row.get("title") else None,
        "text": text if isinstance(text, str) else None,
        "normalized_text": normalize_text(text),
        "language": None,
        "timestamp": int(row["timestamp"]),
        "source_text_sha256": hashlib.sha256((text or "").encode()).hexdigest(),
        "duplicate_group": text_duplicate_group(text, item_id),
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_review_stages(
    category: str,
    review_url: str,
    metadata: dict[str, dict[str, Any]],
    interactions_path: Path,
    texts_path: Path,
    window_bytes: int,
    batch_size: int,
    provenance: dict[str, Any],
) -> tuple[int, int, int]:
    """Scan reviews once and write both normalized review tables."""

    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    interaction_writer = pq.ParquetWriter(
        interactions_path, INTERACTION_SCHEMA, compression="zstd", version="2.6"
    )
    text_writer = pq.ParquetWriter(
        texts_path, REVIEW_TEXT_SCHEMA, compression="zstd", version="2.6"
    )
    interaction_batch: list[dict[str, Any]] = []
    text_batch: list[dict[str, Any]] = []
    raw_count = 0
    scoped_count = 0
    try:
        for row in iter_jsonl_records(review_url, chunk_bytes=window_bytes, provenance=provenance):
            raw_count += 1
            item_id = row.get("parent_asin") or row.get("asin")
            if item_id not in metadata:
                continue
            scoped_count += 1
            interaction_batch.append(interaction_row(category, row))
            text_batch.append(review_text_row(category, row))
            if len(interaction_batch) >= batch_size:
                interaction_writer.write_table(
                    pa.Table.from_pylist(interaction_batch, schema=INTERACTION_SCHEMA)
                )
                text_writer.write_table(pa.Table.from_pylist(text_batch, schema=REVIEW_TEXT_SCHEMA))
                interaction_batch.clear()
                text_batch.clear()
        if interaction_batch:
            interaction_writer.write_table(
                pa.Table.from_pylist(interaction_batch, schema=INTERACTION_SCHEMA)
            )
            text_writer.write_table(pa.Table.from_pylist(text_batch, schema=REVIEW_TEXT_SCHEMA))
    finally:
        interaction_writer.close()
        text_writer.close()
    return raw_count, scoped_count, provenance.get("source_bytes", 0)


def sql_identifier(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def materialize_tables(
    temp_dir: Path,
    snapshot_dir: Path,
    cutoff_t0: int,
    cutoff_t1: int,
) -> dict[str, int]:
    """Deduplicate and derive time-safe tables with DuckDB."""

    raw_interactions = sql_identifier(temp_dir / "raw_interactions.parquet")
    raw_texts = sql_identifier(temp_dir / "raw_review_texts.parquet")
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE interactions AS
            SELECT * EXCLUDE (row_number) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY review_id
                    ORDER BY timestamp, user_id, item_id, review_id
                ) AS row_number
                FROM read_parquet({raw_interactions})
            ) WHERE row_number = 1
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE review_texts AS
            SELECT * EXCLUDE (row_number) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY review_id
                    ORDER BY timestamp, item_id, review_id
                ) AS row_number
                FROM read_parquet({raw_texts})
            ) WHERE row_number = 1
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE first_pair AS
            SELECT * EXCLUDE (row_number) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY user_id, item_id
                    ORDER BY timestamp, review_id
                ) AS row_number
                FROM interactions
            ) WHERE row_number = 1
            """
        )
        queries = {
            "interactions": (
                "SELECT * FROM interactions ORDER BY timestamp, user_id, item_id, review_id"
            ),
            "review_texts": ("SELECT * FROM review_texts ORDER BY timestamp, item_id, review_id"),
            "train_interactions": (
                f"SELECT * FROM interactions WHERE timestamp < {cutoff_t0} "
                "ORDER BY timestamp, user_id, item_id, review_id"
            ),
            "fit_interactions": (
                f"SELECT * FROM interactions WHERE timestamp < {cutoff_t1} "
                "ORDER BY timestamp, user_id, item_id, review_id"
            ),
            "validation_targets": f"""
                SELECT p.* FROM first_pair p
                WHERE p.timestamp >= {cutoff_t0} AND p.timestamp < {cutoff_t1}
                  AND p.rating >= 4
                  AND EXISTS (
                      SELECT 1 FROM interactions h
                      WHERE h.timestamp < {cutoff_t0} AND h.item_id = p.item_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM interactions h
                      WHERE h.timestamp < {cutoff_t0}
                        AND h.user_id = p.user_id AND h.item_id = p.item_id
                  )
                ORDER BY p.timestamp, p.user_id, p.item_id, p.review_id
            """,
            "test_targets": f"""
                SELECT p.* FROM first_pair p
                WHERE p.timestamp >= {cutoff_t1}
                  AND p.rating >= 4
                  AND EXISTS (
                      SELECT 1 FROM interactions h
                      WHERE h.timestamp < {cutoff_t1} AND h.item_id = p.item_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM interactions h
                      WHERE h.timestamp < {cutoff_t1}
                        AND h.user_id = p.user_id AND h.item_id = p.item_id
                  )
                ORDER BY p.timestamp, p.user_id, p.item_id, p.review_id
            """,
        }
        counts: dict[str, int] = {}
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name, query in queries.items():
            output = snapshot_dir / f"{name}.parquet"
            connection.execute(
                f"COPY ({query}) TO {sql_identifier(output)} (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            counts[name] = int(connection.execute(f"SELECT count(*) FROM ({query})").fetchone()[0])
        return counts
    finally:
        connection.close()


def build_full_snapshot(
    category: str,
    output_root: Path,
    manifest_dir: Path,
    window_bytes: int,
    batch_size: int,
    excluded_terms: tuple[str, ...] = EXCLUDED_ENTITY_TERMS,
) -> dict[str, Any]:
    review_url = f"{BASE_URL}/raw/review_categories/{category}.jsonl"
    metadata_url = f"{BASE_URL}/raw/meta_categories/meta_{category}.jsonl"
    metadata_provenance: dict[str, Any] = {}
    scoped_metadata: dict[str, dict[str, Any]] = {}
    metadata_rows = 0
    metadata_missing = Counter()
    metadata_fields = (
        "main_category",
        "title",
        "description",
        "features",
        "price",
        "brand",
        "images",
        "details",
    )
    for row in iter_jsonl_records(
        metadata_url, chunk_bytes=window_bytes, provenance=metadata_provenance
    ):
        metadata_rows += 1
        for field in metadata_fields:
            value = row.get(field)
            if value is None or value == "" or value == []:
                metadata_missing[field] += 1
        item_id = row.get("parent_asin") or row.get("asin")
        if item_id and is_software_game(row, excluded_terms):
            scoped_metadata[str(item_id)] = row
    if not scoped_metadata:
        raise ValueError("full metadata scan found no software-game items")

    temp_root = Path(tempfile.mkdtemp(prefix=f"trustrec-{category.lower()}-"))
    try:
        raw_interactions = temp_root / "raw_interactions.parquet"
        raw_texts = temp_root / "raw_review_texts.parquet"

        review_provenance: dict[str, Any] = {}
        raw_review_count, interaction_count, _ = write_review_stages(
            category,
            review_url,
            scoped_metadata,
            raw_interactions,
            raw_texts,
            window_bytes,
            batch_size,
            review_provenance,
        )

        connection = duckdb.connect()
        try:
            quantiles = connection.execute(
                "SELECT quantile_cont(timestamp, 0.80), "
                "quantile_cont(timestamp, 0.90) FROM ("
                "SELECT timestamp, row_number() OVER ("
                "PARTITION BY review_id ORDER BY timestamp, user_id, item_id, review_id"
                ") AS row_number "
                f"FROM read_parquet({sql_identifier(raw_interactions)})"
                ") WHERE row_number = 1"
            ).fetchone()
        finally:
            connection.close()
        cutoff_t0, cutoff_t1 = round(quantiles[0]), round(quantiles[1])
        if cutoff_t0 >= cutoff_t1:
            raise ValueError("full source does not provide distinct T0 and T1 cutoffs")

        connection = duckdb.connect()
        try:
            review_stats = connection.execute(
                f"""
                SELECT
                    count(*) AS rows,
                    count(*) FILTER (WHERE text IS NULL OR trim(text) = '') AS text_missing,
                    avg(length(text)) FILTER (
                        WHERE text IS NOT NULL AND trim(text) <> ''
                    ) AS text_mean
                FROM read_parquet({sql_identifier(raw_texts)})
                """
            ).fetchone()
            rating_stats = connection.execute(
                "SELECT rating, count(*) "
                f"FROM read_parquet({sql_identifier(raw_interactions)}) "
                "GROUP BY rating ORDER BY rating"
            ).fetchall()
        finally:
            connection.close()

        snapshot_seed = hashlib.sha256(
            canonical_json(
                {
                    "category": category,
                    "review_source_sha256": review_provenance["source_sha256"],
                    "metadata_source_sha256": metadata_provenance["source_sha256"],
                    "cutoff_t0": cutoff_t0,
                    "cutoff_t1": cutoff_t1,
                    "scope": "software_games_only_after_metadata_join",
                }
            )
        ).hexdigest()
        snapshot_id = f"{category.lower()}-full-{snapshot_seed[:12]}"
        snapshot_dir = output_root / category.lower() / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        counts = materialize_tables(temp_root, snapshot_dir, cutoff_t0, cutoff_t1)
        metadata_path = snapshot_dir / "item_metadata.parquet"
        metadata_count = write_batches(
            metadata_path,
            METADATA_SCHEMA,
            (
                metadata_row(item_id, scoped_metadata[item_id])
                for item_id in sorted(scoped_metadata)
            ),
            batch_size,
        )
        counts["item_metadata"] = metadata_count
        artifact_paths = {
            name: snapshot_dir / f"{name}.parquet"
            for name in (
                "interactions",
                "review_texts",
                "item_metadata",
                "train_interactions",
                "fit_interactions",
                "validation_targets",
                "test_targets",
            )
        }
        artifact_hashes = {name: hash_file(path) for name, path in artifact_paths.items()}
        connection = duckdb.connect()
        try:
            unique_users, unique_items = connection.execute(
                "SELECT count(DISTINCT user_id), count(DISTINCT item_id) "
                f"FROM read_parquet({sql_identifier(artifact_paths['interactions'])})"
            ).fetchone()
            validation_users = connection.execute(
                "SELECT count(DISTINCT user_id) "
                f"FROM read_parquet({sql_identifier(artifact_paths['validation_targets'])})"
            ).fetchone()[0]
            test_users = connection.execute(
                "SELECT count(DISTINCT user_id) "
                f"FROM read_parquet({sql_identifier(artifact_paths['test_targets'])})"
            ).fetchone()[0]
        finally:
            connection.close()
        dataset_hash = hashlib.sha256(
            canonical_json(
                {
                    "artifact_sha256": artifact_hashes,
                    "cutoffs": {"t0": cutoff_t0, "t1": cutoff_t1},
                    "metadata_source_sha256": metadata_provenance["source_sha256"],
                    "review_source_sha256": review_provenance["source_sha256"],
                    "scope": "software_games_only_after_metadata_join",
                }
            )
        ).hexdigest()
        manifest = {
            "schema_version": 3,
            "snapshot_id": snapshot_id,
            "category": category,
            "artifact_role": "recommendation_benchmark",
            "benchmark_eligible": True,
            "entity_scope_status": "software_games_only_after_metadata_join",
            "metadata_usage": "scope_and_display_only",
            "created_at": datetime.now(UTC).isoformat(),
            "profile_type": "full-category-stream",
            "cutoffs": {
                "t0": datetime.fromtimestamp(cutoff_t0 / 1000, UTC).isoformat(),
                "t1": datetime.fromtimestamp(cutoff_t1 / 1000, UTC).isoformat(),
                "policy": "global_timestamp_quantiles_0.80_0.90",
            },
            "dataset_hash": dataset_hash,
            "row_counts": counts,
            "unique_counts": {
                "users": int(unique_users),
                "items": int(unique_items),
                "metadata_matched_items": int(unique_items),
                "validation_target_users": int(validation_users),
                "test_target_users": int(test_users),
            },
            "metadata_coverage": 1.0,
            "sources": {"reviews": review_provenance, "metadata": metadata_provenance},
            "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
            "artifact_sha256": artifact_hashes,
            "sampling": {
                "mode": "full_sequential_range_scan",
                "window_bytes": window_bytes,
                "batch_size": batch_size,
                "selection": (
                    "deduplicate_by_stable_review_id_then_sort_by_timestamp_user_item_review"
                ),
            },
            "scope_counts": {
                "reviews_scanned": raw_review_count,
                "metadata_rows_scanned": metadata_rows,
                "software_game_items": len(scoped_metadata),
                "software_game_reviews_before_deduplication": interaction_count,
                "excluded_reviews": raw_review_count - interaction_count,
            },
            "review_profile": {
                "rows": int(review_stats[0]),
                "text_missing": int(review_stats[1]),
                "text_missing_fraction": round(review_stats[1] / review_stats[0], 6)
                if review_stats[0]
                else 0.0,
                "mean_non_empty_text_chars": round(float(review_stats[2] or 0.0), 2),
                "rating_distribution": {str(rating): count for rating, count in rating_stats},
            },
            "metadata_profile": {
                "rows": metadata_rows,
                "missingness": {
                    field: round(metadata_missing[field] / metadata_rows, 6)
                    if metadata_rows
                    else 0.0
                    for field in metadata_fields
                },
            },
        }
        manifest_path = manifest_dir / f"{snapshot_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--window-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    manifest = build_full_snapshot(
        category=args.category,
        output_root=args.output_root,
        manifest_dir=args.manifest_dir,
        window_bytes=args.window_bytes,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
