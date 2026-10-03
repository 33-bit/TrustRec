"""Create a reviewer-friendly CSV from Laya JSONL labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unit_id",
        "review_id",
        "item_id",
        "timestamp",
        "text",
        "labels_json",
        "out_of_scope",
        "needs_adjudication",
        "review_reasons",
        "presence_probability_json",
        "confidence_json",
        "human_labels_json",
        "human_note",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "unit_id": row["unit_id"],
                    "review_id": row["review_id"],
                    "item_id": row["item_id"],
                    "timestamp": row["timestamp"],
                    "text": row["text"],
                    "labels_json": json.dumps(row["labels"], ensure_ascii=False),
                    "out_of_scope": row["out_of_scope"],
                    "needs_adjudication": row["needs_adjudication"],
                    "review_reasons": ";".join(row.get("review_reasons", [])),
                    "presence_probability_json": json.dumps(
                        row.get("presence_probability", {}), ensure_ascii=False
                    ),
                    "confidence_json": json.dumps(row.get("confidence", {}), ensure_ascii=False),
                    "human_labels_json": "",
                    "human_note": "",
                }
            )
    print(f"Wrote {len(rows)} review rows to {args.output}")


if __name__ == "__main__":
    main()
