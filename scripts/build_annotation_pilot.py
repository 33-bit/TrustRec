"""Create a deterministic 100-review aspect annotation worksheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

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
    lowered = text.lower()
    return [
        aspect
        for aspect, keywords in ASPECT_KEYWORDS.items()
        if any(word in lowered for word in keywords)
    ]


def choose_rows(rows: list[dict], limit: int = 100) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rating = int(round(float(row.get("rating") or 0)))
        if rating in RATING_QUOTAS and row.get("text"):
            buckets[rating].append(row)
    selected = []
    for rating, quota in RATING_QUOTAS.items():
        candidates = sorted(
            buckets[rating],
            key=lambda row: hashlib.sha256(str(row["review_id"]).encode()).hexdigest(),
        )
        selected.extend(candidates[:quota])
    if len(selected) < limit:
        used = {row["review_id"] for row in selected}
        remaining = sorted(
            (row for row in rows if row.get("text") and row["review_id"] not in used),
            key=lambda row: hashlib.sha256(str(row["review_id"]).encode()).hexdigest(),
        )
        selected.extend(remaining[: limit - len(selected)])
    return sorted(selected[:limit], key=lambda row: row["review_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/tasks/t2_1_annotation_pilot.csv"))
    args = parser.parse_args()
    rows = parquet.read_table(args.snapshot / "review_texts.parquet").to_pylist()
    interactions = {
        row["review_id"]: row
        for row in parquet.read_table(args.snapshot / "interactions.parquet").to_pylist()
    }
    joined = [
        {
            **row,
            **{
                key: interactions[row["review_id"]][key]
                for key in ("user_id", "rating", "timestamp")
            },
        }
        for row in rows
        if row["review_id"] in interactions
    ]
    selected = choose_rows(joined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotation_id",
        "review_id",
        "item_id",
        "rating",
        "timestamp",
        "text",
        "weak_aspect_suggestions",
        "annotator_1_labels_json",
        "annotator_2_labels_json",
        "adjudicated_labels_json",
        "adjudication_note",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "annotation_id": f"t2.1-{index:03d}",
                    "review_id": row["review_id"],
                    "item_id": row["item_id"],
                    "rating": row["rating"],
                    "timestamp": row["timestamp"],
                    "text": row["text"],
                    "weak_aspect_suggestions": ",".join(weak_aspect_suggestions(row["text"])),
                    "annotator_1_labels_json": "",
                    "annotator_2_labels_json": "",
                    "adjudicated_labels_json": "",
                    "adjudication_note": "",
                }
            )
    print(f"Wrote {len(selected)} annotation units to {args.output}")


if __name__ == "__main__":
    main()
