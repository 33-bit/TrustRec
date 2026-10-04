"""Build the frozen source split for the LLM pseudo-test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIMIT = 100
DEFAULT_RATING_QUOTA = 20
DEFAULT_MINIMUM_TEXT_LENGTH = 100
DEFAULT_SELECTION_SEED = 2202


def parse_cutoff(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("the cutoff must include a time zone")
    return round(timestamp.timestamp() * 1000)


def select_rows(
    review_rows: list[dict[str, Any]],
    interaction_rows: list[dict[str, Any]],
    *,
    pilot_review_ids: set[str],
    pilot_group_keys: set[tuple[str, str]],
    cutoff_timestamp_ms: int,
    limit: int = DEFAULT_LIMIT,
    rating_quota: int = DEFAULT_RATING_QUOTA,
    minimum_text_length: int = DEFAULT_MINIMUM_TEXT_LENGTH,
    require_exact_rating_quota: bool = False,
) -> list[dict[str, Any]]:
    """Select deterministic, rating-stratified rows outside the development pilot."""

    if limit <= 0 or rating_quota <= 0:
        raise ValueError("limit and rating_quota must be positive")
    interactions = {str(row["review_id"]): row for row in interaction_rows}
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for review in review_rows:
        review_id = str(review.get("review_id", ""))
        text = review.get("text")
        group_key = (str(review.get("item_id", "")), str(review.get("duplicate_group", "")))
        if not review_id or review_id in pilot_review_ids or group_key in pilot_group_keys:
            continue
        if not isinstance(text, str) or len(text.strip()) < minimum_text_length:
            continue
        timestamp = int(review.get("timestamp", 0))
        if timestamp >= cutoff_timestamp_ms:
            continue
        interaction = interactions.get(review_id)
        if interaction is None:
            continue
        rating_value = float(interaction.get("rating", 0))
        if not rating_value.is_integer():
            continue
        rating = int(rating_value)
        if rating not in range(1, 6):
            continue
        buckets[rating].append(
            {
                **review,
                "review_id": review_id,
                "user_id": interaction.get("user_id"),
                "rating": interaction.get("rating"),
                "timestamp": timestamp,
            }
        )

    selected: list[dict[str, Any]] = []
    if require_exact_rating_quota and (
        limit != DEFAULT_LIMIT or rating_quota != DEFAULT_RATING_QUOTA
    ):
        raise ValueError("strict T2.2 selection requires 100 rows and 20 rows per rating")
    if require_exact_rating_quota and any(
        len(buckets[rating]) < rating_quota for rating in range(1, 6)
    ):
        counts = {rating: len(buckets[rating]) for rating in range(1, 6)}
        raise ValueError(f"each rating bucket needs {rating_quota} rows, found {counts}")
    for rating in range(1, 6):
        candidates = sorted(
            buckets[rating], key=lambda row: (-int(row["timestamp"]), row["review_id"])
        )
        selected.extend(candidates[:rating_quota])
    if len(selected) < limit:
        used = {row["review_id"] for row in selected}
        remaining = sorted(
            (
                row
                for candidates in buckets.values()
                for row in candidates
                if row["review_id"] not in used
            ),
            key=lambda row: (-int(row["timestamp"]), row["review_id"]),
        )
        selected.extend(remaining[: limit - len(selected)])
    if len(selected) < limit:
        raise ValueError(
            f"only {len(selected)} eligible rows are available, but {limit} are required"
        )
    return sorted(selected[:limit], key=lambda row: (int(row["timestamp"]), str(row["review_id"])))


def _resolve(path_value: str, base: Path = ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("snapshot_id") or not manifest.get("dataset_hash"):
        raise ValueError("source snapshot manifest lacks snapshot_id or dataset_hash")
    cutoff = manifest.get("cutoffs", {}).get("t0")
    if not cutoff:
        raise ValueError("source snapshot manifest lacks cutoffs.t0")
    return manifest


def _load_pilot(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_selection(
    snapshot_manifest_path: Path,
    pilot_jsonl_path: Path,
    *,
    limit: int = DEFAULT_LIMIT,
    rating_quota: int = DEFAULT_RATING_QUOTA,
    minimum_text_length: int = DEFAULT_MINIMUM_TEXT_LENGTH,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    require_exact_rating_quota: bool = True,
) -> list[dict[str, Any]]:
    """Read the source snapshot and return the frozen split source rows."""

    manifest = _load_manifest(snapshot_manifest_path)
    pilot = _load_pilot(pilot_jsonl_path)
    source_path = _resolve(manifest["artifact_paths"]["review_texts"])
    interaction_path = _resolve(manifest["artifact_paths"]["interactions"])
    connection = duckdb.connect()
    connection.execute("create temp table pilot_ids(review_id varchar)")
    connection.executemany(
        "insert into pilot_ids values (?)", [(row["review_id"],) for row in pilot]
    )
    connection.execute(
        """
        create temp table pilot_groups as
        select distinct r.item_id, r.duplicate_group
        from read_parquet(?) r
        join pilot_ids p using (review_id)
        """,
        [str(source_path)],
    )
    rows = connection.execute(
        """
        select r.review_id, r.item_id, r.text, r.timestamp, r.duplicate_group,
               i.user_id, i.rating
        from read_parquet(?) r
        join read_parquet(?) i using (review_id)
        where r.timestamp < ?
          and r.text is not null
          and not exists (select 1 from pilot_ids p where p.review_id = r.review_id)
            and not exists (
                select 1 from pilot_groups g
                where g.item_id is not distinct from r.item_id
                  and g.duplicate_group is not distinct from r.duplicate_group
            )
        """,
        [str(source_path), str(interaction_path), parse_cutoff(manifest["cutoffs"]["t0"])],
    ).fetchdf()
    selected = select_rows(
        rows.to_dict(orient="records"),
        rows[["review_id", "user_id", "rating", "timestamp"]].to_dict(orient="records"),
        pilot_review_ids=set(),
        pilot_group_keys=set(),
        cutoff_timestamp_ms=parse_cutoff(manifest["cutoffs"]["t0"]),
        limit=limit,
        rating_quota=rating_quota,
        minimum_text_length=minimum_text_length,
        require_exact_rating_quota=require_exact_rating_quota,
    )
    for row in selected:
        row["source_text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
        row["selection_seed"] = selection_seed
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--pilot-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--rating-quota", type=int, default=DEFAULT_RATING_QUOTA)
    parser.add_argument("--minimum-text-length", type=int, default=DEFAULT_MINIMUM_TEXT_LENGTH)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args()
    if (
        args.limit != DEFAULT_LIMIT
        or args.rating_quota != DEFAULT_RATING_QUOTA
        or args.minimum_text_length != DEFAULT_MINIMUM_TEXT_LENGTH
        or args.selection_seed != DEFAULT_SELECTION_SEED
    ):
        raise SystemExit(
            "T2.2 requires limit=100, rating-quota=20, "
            "minimum-text-length=100, and selection-seed=2202"
        )
    rows = build_selection(
        args.snapshot_manifest,
        args.pilot_jsonl,
        limit=args.limit,
        rating_quota=args.rating_quota,
        minimum_text_length=args.minimum_text_length,
        selection_seed=args.selection_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            json.dumps(
                {
                    "selection_rank": index,
                    "review_id": row["review_id"],
                    "item_id": row["item_id"],
                    "duplicate_group": row["duplicate_group"],
                    "user_id": "user-"
                    + hashlib.sha256(str(row["user_id"]).encode()).hexdigest()[:16],
                    "rating": row["rating"],
                    "timestamp": row["timestamp"],
                    "source_text_sha256": row["source_text_sha256"],
                    "selection_seed": row["selection_seed"],
                },
                sort_keys=True,
            )
            for index, row in enumerate(rows, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} LLM pseudo-test source units to {args.output}")


if __name__ == "__main__":
    main()
