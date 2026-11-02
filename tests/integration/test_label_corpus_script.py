import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.build_evidence_profiles import build_profiles
from scripts.label_corpus_evidence import label_corpus

CUTOFF = "1970-01-01T00:00:00.100000+00:00"
TEXTS = (
    "The story is great. The controls are terrible.",
    "Graphics look beautiful but the game crashes constantly.",
    "Fun gameplay and a lot of content to replay.",
    "Not worth the price.",
)


def _write_snapshot(tmp_path: Path) -> Path:
    interactions, texts = [], []
    for position in range(40):
        review_id = f"r{position:02d}"
        text = TEXTS[position % len(TEXTS)]
        timestamp = position * 4
        texts.append(
            {
                "review_id": review_id,
                "item_id": f"i{position % 5}",
                "text": text,
                "timestamp": timestamp,
                "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "duplicate_group": f"g{position}",
            }
        )
        if timestamp < 100 and position != 3:
            interactions.append(
                {
                    "review_id": review_id,
                    "user_id": f"u{position % 7}",
                    "item_id": f"i{position % 5}",
                    "rating": 5.0,
                    "timestamp": timestamp,
                }
            )
    pq.write_table(pa.Table.from_pylist(interactions), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(texts), tmp_path / "texts.parquet")
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "snap-1",
                "dataset_hash": "d" * 64,
                "cutoffs": {"t0": CUTOFF},
                "artifact_paths": {
                    "train_interactions": "train.parquet",
                    "review_texts": "texts.parquet",
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_label_corpus_feeds_evidence_profiles(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    output = tmp_path / "evidence.jsonl"
    run_manifest = label_corpus(manifest, cutoff_name="t0", output_path=output, batch_size=7)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert run_manifest["status"] == "complete"
    assert run_manifest["summary"]["reviews"] == 24
    assert rows
    assert all(row["timestamp"] < 100 for row in rows)
    assert "r03" not in {row["review_id"] for row in rows}
    profiles = build_profiles(
        output,
        snapshot_id="snap-1",
        dataset_hash="d" * 64,
        cutoff_timestamp=CUTOFF,
        output_path=tmp_path / "profiles.json",
    )
    assert profiles["profiles"]


def test_label_corpus_workers_match_a_single_process(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    single = tmp_path / "single.jsonl"
    forked = tmp_path / "forked.jsonl"
    label_corpus(manifest, cutoff_name="t0", output_path=single, batch_size=5)
    label_corpus(manifest, cutoff_name="t0", output_path=forked, batch_size=5, workers=2)

    assert single.read_bytes() == forked.read_bytes()


def test_label_corpus_refuses_to_replace_a_completed_manifest(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    output = tmp_path / "evidence.jsonl"
    label_corpus(manifest, cutoff_name="t0", output_path=output)

    with pytest.raises(FileExistsError, match="completed"):
        label_corpus(manifest, cutoff_name="t0", output_path=output)
