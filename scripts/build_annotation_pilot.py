"""Build a deterministic LLM development pilot from a time-safe snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

RATING_QUOTAS = {1: 20, 2: 15, 3: 15, 4: 25, 5: 25}
ASPECT_KEYWORDS = {
    "gameplay": ("gameplay", "combat", "mechanic", "level", "fun"),
    "story": ("story", "plot", "character", "narrative", "dialogue"),
    "graphics": ("graphic", "visual", "animation", "art", "beautiful"),
    "performance": ("crash", "lag", "bug", "slow", "performance", "glitch"),
    "controls": ("control", "controller", "button", "camera", "responsive"),
    "multiplayer": ("multiplayer", "online", "coop", "co-op", "matchmaking"),
    "content_replay": ("content", "replay", "hours", "endgame", "repeat"),
    "value": ("worth", "value", "money", "price", "buy", "cost"),
}


def weak_aspect_suggestions(text: str) -> list[str]:
    """Return optional keyword hints for pilot triage."""

    lowered = text.casefold()
    return [
        aspect
        for aspect, keywords in ASPECT_KEYWORDS.items()
        if any(word in lowered for word in keywords)
    ]


def parse_cutoff(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("the cutoff must include a time zone")
    return round(timestamp.timestamp() * 1000)


def choose_rows(rows: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    """Select rating quotas in deterministic source order."""

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rating = int(round(float(row.get("rating") or 0)))
        if rating in RATING_QUOTAS and row.get("text"):
            buckets[rating].append(row)
    selected: list[dict[str, Any]] = []
    for rating, quota in RATING_QUOTAS.items():
        candidates = sorted(
            buckets[rating],
            key=lambda row: (
                int(row["timestamp"]),
                str(row["user_id"]),
                str(row["item_id"]),
                str(row["review_id"]),
            ),
        )
        selected.extend(candidates[:quota])
    if len(selected) < limit:
        used = {row["review_id"] for row in selected}
        remaining = sorted(
            (row for row in rows if row.get("text") and row["review_id"] not in used),
            key=lambda row: (
                int(row["timestamp"]),
                str(row["user_id"]),
                str(row["item_id"]),
                str(row["review_id"]),
            ),
        )
        selected.extend(remaining[: limit - len(selected)])
    if len(selected) < limit:
        raise ValueError(
            f"the snapshot has only {len(selected)} eligible text reviews, but {limit} are required"
        )
    return sorted(
        selected[:limit],
        key=lambda row: (
            int(row["timestamp"]),
            str(row["user_id"]),
            str(row["item_id"]),
            str(row["review_id"]),
        ),
    )


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    try:
        source_cutoff = manifest["cutoffs"]["t0"]
        snapshot_id = manifest["snapshot_id"]
    except KeyError as error:
        raise ValueError(f"snapshot manifest lacks {error.args[0]}") from error
    manifest["source_cutoff_timestamp"] = source_cutoff
    manifest["snapshot_id"] = snapshot_id
    return manifest


def resolve_snapshot_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    artifact_path = Path(manifest["artifact_paths"]["review_texts"])
    if not artifact_path.is_absolute():
        artifact_path = manifest_path.parent.parent.parent / artifact_path
    return artifact_path.parent


def build_pilot_rows(snapshot: Path, snapshot_id: str, cutoff: str) -> list[dict[str, Any]]:
    cutoff_ms = parse_cutoff(cutoff)
    texts = parquet.read_table(snapshot / "review_texts.parquet").to_pylist()
    interactions = {
        row["review_id"]: row
        for row in parquet.read_table(snapshot / "interactions.parquet").to_pylist()
    }
    joined = []
    for row in texts:
        interaction = interactions.get(row["review_id"])
        if interaction is None:
            continue
        if int(interaction["timestamp"]) >= cutoff_ms:
            continue
        if not interaction.get("user_id") or not interaction.get("item_id"):
            raise ValueError(f"review {row['review_id']} lacks a required ID")
        joined.append(
            {
                **row,
                "user_id": interaction["user_id"],
                "rating": interaction["rating"],
                "timestamp": int(interaction["timestamp"]),
            }
        )
    selected = choose_rows(joined)
    rows = []
    for index, row in enumerate(selected, start=1):
        source_hash = hashlib.sha256((row.get("text") or "").encode()).hexdigest()
        rows.append(
            {
                "schema_version": 2,
                "unit_id": f"t2.1-pilot-{index:03d}",
                "review_id": row["review_id"],
                "item_id": row["item_id"],
                "user_id": "user-" + hashlib.sha256(str(row["user_id"]).encode()).hexdigest()[:16],
                "rating": row["rating"],
                "timestamp": row["timestamp"],
                "source_snapshot_id": snapshot_id,
                "source_cutoff_timestamp": cutoff,
                "split_role": "development_pilot",
                "label_status": "llm_silver",
                "text": row["text"],
                "source_text_sha256": source_hash,
                "weak_aspect_suggestions": ",".join(weak_aspect_suggestions(row["text"])),
                "usage_restrictions": json.dumps(
                    {
                        "allowed_uses": ["development", "optional_nlp_training"],
                        "prohibited_uses": ["recommendation_test", "llm_pseudo_test_scoring"],
                    },
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot directory")
    parser.add_argument(
        "--manifest",
        "--snapshot-manifest",
        dest="manifest",
        type=Path,
        help="Snapshot manifest with an exact t0 timestamp",
    )
    parser.add_argument("--cutoff", help="Exact ISO-8601 cutoff when no manifest is supplied")
    parser.add_argument("--snapshot-id", help="Snapshot ID when no manifest is supplied")
    parser.add_argument("--output", type=Path, default=Path("docs/tasks/t2_1_annotation_pilot.csv"))
    args = parser.parse_args()

    if args.manifest:
        manifest = load_manifest(args.manifest)
        snapshot = args.snapshot or resolve_snapshot_path(manifest, args.manifest)
        cutoff = args.cutoff or manifest["source_cutoff_timestamp"]
        snapshot_id = args.snapshot_id or manifest["snapshot_id"]
    else:
        if not args.snapshot or not args.cutoff:
            raise SystemExit("provide --manifest or --snapshot and --cutoff")
        snapshot = args.snapshot
        cutoff = args.cutoff
        snapshot_id = args.snapshot_id or snapshot.name

    rows = build_pilot_rows(snapshot, snapshot_id, cutoff)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} LLM development-pilot units to {args.output}")


if __name__ == "__main__":
    main()
