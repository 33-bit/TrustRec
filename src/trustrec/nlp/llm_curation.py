"""Consolidate a raw LLM label cache into clean, deduplicated Parquet tables.

The raw cache is an append log: one JSON line per labeling attempt, with failed
attempts, retries, and per-row run metadata. Curation keeps one record per
review, checks it against the snapshot text, applies a few fixed cleaning rules
to the labels, and writes two tables:

- ``reviews.parquet``: one row per labeled review with its status and flags.
- ``labels.parquet``: one row per aspect label with exact source offsets.

Neither table holds reviewer or item identities. Join on ``review_id`` with the
snapshot to get them.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .llm_labeler import PROMPT_VERSION, cache_files, prompt_hash
from .records import ASPECTS, SENTIMENTS

CURATED_SCHEMA_VERSION = 1
CLEANING_VERSION = "llm-curation-v1"
REVIEWS_FILE = "reviews.parquet"
LABELS_FILE = "labels.parquet"
MANIFEST_FILE = "manifest.json"
REVIEW_STATUSES = ("labeled", "no_aspect", "out_of_scope", "empty_text")

REVIEWS_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("source_text_sha256", pa.string()),
        ("status", pa.string()),
        ("needs_review", pa.bool_()),
        ("truncated", pa.bool_()),
        ("n_labels", pa.int32()),
        ("dropped_labels", pa.int32()),
    ]
)
LABELS_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("label_index", pa.int32()),
        ("aspect", pa.string()),
        ("polarity", pa.string()),
        ("start", pa.int32()),
        ("end", pa.int32()),
        ("evidence", pa.string()),
    ]
)


@dataclass
class CurationStats:
    """Counts of records and labels that curation dropped or changed."""

    raw_records: int = 0
    invalid_json_lines: int = 0
    error_records: int = 0
    other_setting_records: int = 0
    superseded_records: int = 0
    out_of_scope_ids: int = 0
    text_changed: int = 0
    raw_labels: int = 0
    labels: int = 0
    actions: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_records": self.raw_records,
            "invalid_json_lines": self.invalid_json_lines,
            "error_records": self.error_records,
            "other_setting_records": self.other_setting_records,
            "superseded_records": self.superseded_records,
            "reviews_not_in_scope": self.out_of_scope_ids,
            "text_changed": self.text_changed,
            "raw_labels": self.raw_labels,
            "labels": self.labels,
            "label_actions": dict(sorted(self.actions.items())),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collapse_cache(
    cache_path: Path,
    model_id: str,
    reasoning_effort: str | None,
    stats: CurationStats,
) -> dict[str, dict[str, Any]]:
    """Return the last successful record per review for one model setting.

    A later successful attempt replaces an earlier one. Failed attempts and
    records from another model, reasoning setting, or prompt are counted and
    skipped.
    """

    records: dict[str, dict[str, Any]] = {}
    for part in cache_files(cache_path):
        with part.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.invalid_json_lines += 1
                    continue
                stats.raw_records += 1
                if record.get("error"):
                    stats.error_records += 1
                    continue
                if (
                    record.get("prompt_version") != PROMPT_VERSION
                    or record.get("model_id") != model_id
                    or record.get("reasoning_effort") != reasoning_effort
                ):
                    stats.other_setting_records += 1
                    continue
                if record["review_id"] in records:
                    stats.superseded_records += 1
                records[record["review_id"]] = record
    return records


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def clean_labels(
    text: str, labels: list[Mapping[str, Any]], actions: Counter[str]
) -> list[dict[str, Any]]:
    """Apply the curation rules to one review's labels and return them sorted.

    1. Evidence must equal ``text[start:end]``; a mismatch is an error.
    2. Leading and trailing whitespace is trimmed from each span.
    3. One aspect with both ``positive`` and ``negative`` on the same span becomes
       one ``neutral`` label, because the claim is mixed.
    4. Overlapping spans with the same aspect and polarity merge into their union,
       so one claim is not counted twice.
    5. Labels sort by start, end, aspect, and polarity.
    """

    spans: dict[tuple[str, int, int], set[str]] = {}
    for label in labels:
        start, end = int(label["start"]), int(label["end"])
        if text[start:end] != label["evidence"]:
            raise ValueError("evidence does not match its offsets")
        if label["aspect"] not in ASPECTS or label["polarity"] not in SENTIMENTS:
            raise ValueError(f"unknown aspect or polarity: {label['aspect']}/{label['polarity']}")
        trimmed = _trim(text, start, end)
        if trimmed != (start, end):
            actions["trimmed_whitespace"] += 1
            start, end = trimmed
        if start >= end:
            actions["dropped_empty_span"] += 1
            continue
        key = (label["aspect"], start, end)
        if key in spans and label["polarity"] in spans[key]:
            actions["dropped_exact_duplicate"] += 1
        spans.setdefault(key, set()).add(label["polarity"])

    by_group: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for (aspect, start, end), polarities in spans.items():
        if {"positive", "negative"} <= polarities:
            actions["neutralized_mixed_polarity"] += 1
            polarities = {"neutral"}
        for polarity in polarities:
            by_group.setdefault((aspect, polarity), []).append((start, end))

    cleaned: list[dict[str, Any]] = []
    for (aspect, polarity), ranges in by_group.items():
        ranges = sorted(set(ranges))
        merged: list[list[int]] = []
        for start, end in ranges:
            if merged and start < merged[-1][1]:
                actions["merged_overlap"] += 1
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            cleaned.append(
                {
                    "aspect": aspect,
                    "polarity": polarity,
                    "start": start,
                    "end": end,
                    "evidence": text[start:end],
                }
            )
    cleaned.sort(
        key=lambda label: (label["start"], label["end"], label["aspect"], label["polarity"])
    )
    return cleaned


def review_status(record: Mapping[str, Any], text: str, labels: list[Any]) -> str:
    if not text.strip():
        return "empty_text"
    if record.get("out_of_scope"):
        return "out_of_scope"
    return "labeled" if labels else "no_aspect"


def curate_records(
    records: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    stats: CurationStats,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return review rows and label rows sorted by ``review_id``.

    ``texts`` maps every in-scope review to its source text. Records for other
    reviews, or whose text hash no longer matches, are counted and skipped.
    """

    review_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for review_id in sorted(records):
        record = records[review_id]
        text = texts.get(review_id)
        if text is None:
            stats.out_of_scope_ids += 1
            continue
        if record["source_text_sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest():
            stats.text_changed += 1
            continue
        raw = list(record.get("labels") or [])
        stats.raw_labels += len(raw)
        if record.get("out_of_scope") and raw:
            stats.actions["dropped_out_of_scope_label"] += len(raw)
            raw = []
        try:
            labels = clean_labels(text, raw, stats.actions)
        except ValueError as error:
            raise ValueError(f"review {review_id}: {error}") from error
        status = review_status(record, text, labels)
        review_rows.append(
            {
                "review_id": review_id,
                "source_text_sha256": record["source_text_sha256"],
                "status": status,
                "needs_review": bool(record.get("needs_review")),
                "truncated": bool(record.get("truncated")),
                "n_labels": len(labels),
                "dropped_labels": int(record.get("dropped_labels") or 0),
            }
        )
        for index, label in enumerate(labels):
            label_rows.append({"review_id": review_id, "label_index": index, **label})
        stats.labels += len(labels)
    return review_rows, label_rows


def _write(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> dict[str, Any]:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=250_000)
    return {"rows": table.num_rows, "sha256": sha256_file(path)}


def write_curated(
    output_dir: Path,
    review_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write both tables and a manifest that records counts and file hashes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        REVIEWS_FILE: _write(review_rows, REVIEWS_SCHEMA, output_dir / REVIEWS_FILE),
        LABELS_FILE: _write(label_rows, LABELS_SCHEMA, output_dir / LABELS_FILE),
    }
    statuses = Counter(row["status"] for row in review_rows)
    manifest = {
        **manifest,
        "schema_version": CURATED_SCHEMA_VERSION,
        "cleaning_version": CLEANING_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "counts": {
            "reviews": len(review_rows),
            "labels": len(label_rows),
            "statuses": {status: statuses.get(status, 0) for status in REVIEW_STATUSES},
            "needs_review": sum(row["needs_review"] for row in review_rows),
            "truncated": sum(row["truncated"] for row in review_rows),
            "aspects": dict(sorted(Counter(row["aspect"] for row in label_rows).items())),
            "polarities": dict(sorted(Counter(row["polarity"] for row in label_rows).items())),
        },
        "files": files,
    }
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def is_curated(path: Path) -> bool:
    return path.is_dir() and (path / MANIFEST_FILE).is_file() and (path / LABELS_FILE).is_file()


def read_curated_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CURATED_SCHEMA_VERSION:
        raise ValueError(f"unsupported curated schema: {manifest.get('schema_version')}")
    for name, info in manifest["files"].items():
        if sha256_file(path / name) != info["sha256"]:
            raise ValueError(f"{name} does not match the hash in {MANIFEST_FILE}")
    return manifest


def _iter_table(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=200_000):
        yield from batch.to_pylist()


def load_curated(path: Path) -> dict[str, dict[str, Any]]:
    """Load curated tables as cache-shaped records for ``CachedLLMLabeler``."""

    read_curated_manifest(path)
    records: dict[str, dict[str, Any]] = {}
    for row in _iter_table(path / REVIEWS_FILE, ["review_id", "source_text_sha256", "status"]):
        records[row["review_id"]] = {
            "source_text_sha256": row["source_text_sha256"],
            "out_of_scope": row["status"] == "out_of_scope",
            "labels": [],
        }
    columns = ["review_id", "aspect", "polarity", "start", "end", "evidence"]
    for row in _iter_table(path / LABELS_FILE, columns):
        review_id = row.pop("review_id")
        records[review_id]["labels"].append(row)
    return records
