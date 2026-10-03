"""Reproduce annotation units and validate AI preliminary labels against the source."""

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSHEET = ROOT / "docs/tasks/t2_1_annotation_pilot.csv"
MANIFEST = ROOT / "data/manifests/video_games-pilot-9e665a862c1a.json"
OUTPUT = ROOT / "docs/tasks/t2_1_ai_pilot_labels.jsonl"
ASPECTS = {
    "gameplay",
    "story",
    "graphics",
    "performance",
    "controls",
    "multiplayer",
    "content_replay",
    "value",
}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split raw text without normalization, using Python character offsets."""
    boundary = re.compile(r"<br\s*/?>|(?<=[.!?])\s+|\b(?:but|however|yet)\b", re.IGNORECASE)
    spans = []
    cursor = 0
    for match in boundary.finditer(text):
        # Ellipses and common abbreviations are not sentence boundaries.
        if match.group().isspace() and re.search(
            r"(?:\.\.\.|\b(?:Oct|Feb|Mr|Mrs|Dr|reg|No)\.)$", text[: match.start()]
        ):
            continue
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    result = []
    for start, end in spans:
        prefix = re.match(r"\s*(?:\[\[VIDEOID:[^]]+\]\]\s*)?[-,\s]*", text[start:end])
        start += prefix.end() if prefix else 0
        while end > start and (text[end - 1].isspace() or text[end - 1] == ","):
            end -= 1
        value = text[start:end]
        if not value or value.lower().rstrip(":,.") in {
            "pros",
            "cons",
            "cons: 2",
            "what i liked",
            "what i didn't like",
            "to start with",
            "the good",
            "the bad",
        }:
            continue
        result.append((start, end))
    return result


def select_units() -> tuple[list[dict], int, int]:
    """Select 100 units in rounds over pre-T0 worksheet rows in annotation-ID order."""
    manifest = json.loads(MANIFEST.read_text())
    cutoff_ms = int(datetime.fromisoformat(manifest["cutoffs"]["t0"]).timestamp() * 1000)
    with WORKSHEET.open(newline="") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: row["annotation_id"])
    eligible = [row for row in rows if int(row["timestamp"]) < cutoff_ms]
    candidates = [(row, sentence_spans(row["text"])) for row in eligible]
    units = []
    round_index = 0
    while len(units) < 100:
        added = False
        for row, spans in candidates:
            if round_index >= len(spans):
                continue
            added = True
            start, end = spans[round_index]
            units.append(
                {
                    "unit_id": f"t2.1-ai-v1-{len(units) + 1:03d}",
                    "source_annotation_id": row["annotation_id"],
                    "review_id": row["review_id"],
                    "item_id": row["item_id"],
                    "timestamp": int(row["timestamp"]),
                    "source_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "start": start,
                    "end": end,
                    "text": row["text"][start:end],
                }
            )
            if len(units) == 100:
                break
        if not added:
            raise ValueError("Fewer than 100 nonempty pre-T0 units in the worksheet")
        round_index += 1
    return units, len(eligible), cutoff_ms


def validate(output: Path = OUTPUT) -> None:
    expected, eligible_count, cutoff_ms = select_units()
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    if len(actual) != 100:
        raise ValueError("Expected exactly 100 labels")
    keys = set()
    counts = Counter()
    for want, record in zip(expected, actual, strict=True):
        if any(record.get(key) != value for key, value in want.items()):
            raise ValueError(f"Source/selection mismatch for {want['unit_id']}")
        key = (record["review_id"], record["start"], record["end"])
        if key in keys or record["timestamp"] >= cutoff_ms:
            raise ValueError("Duplicate unit or post-T0 source")
        keys.add(key)
        if record["provenance"]["label_status"] != "ai_preliminary":
            raise ValueError("Labels must be AI preliminary")
        if not isinstance(record["out_of_scope"], bool) or not isinstance(
            record["needs_adjudication"], bool
        ):
            raise ValueError("Expected boolean scope/adjudication flags")
        if record["out_of_scope"] and record["labels"]:
            raise ValueError("Out-of-scope unit must not have ontology labels")
        for label in record["labels"]:
            if label["aspect"] not in ASPECTS or label["polarity"] not in {
                "positive",
                "negative",
                "neutral",
            }:
                raise ValueError("Invalid aspect/polarity")
            start, end = label["start"], label["end"]
            if not record["start"] <= start < end <= record["end"]:
                raise ValueError("Evidence outside selected unit")
            local_start, local_end = start - record["start"], end - record["start"]
            if record["text"][local_start:local_end] != label["evidence"]:
                raise ValueError("Evidence does not match source offsets")
            counts[f"{label['aspect']}/{label['polarity']}"] += 1
    print(
        json.dumps(
            {
                "validated_unique_units": len(keys),
                "eligible_source_reviews": eligible_count,
                "selected_source_reviews": len({r["review_id"] for r in actual}),
                "out_of_scope_units": sum(r["out_of_scope"] for r in actual),
                "needs_adjudication_units": sum(r["needs_adjudication"] for r in actual),
                "empty_label_units": sum(not r["labels"] for r in actual),
                "label_counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    validate(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
