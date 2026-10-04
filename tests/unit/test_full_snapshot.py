import hashlib
import json
from pathlib import Path

from scripts import build_full_snapshot


def test_full_snapshot_streams_and_derives_benchmark_tables(monkeypatch, tmp_path: Path) -> None:
    metadata_rows = [
        {
            "parent_asin": "i1",
            "title": "Game One",
            "main_category": "Video Games",
            "details": {"Type of item": "Video Game"},
        },
        {
            "parent_asin": "i2",
            "title": "Game Two",
            "main_category": "Video Games",
            "details": {"Type of item": "Video Game"},
        },
        {
            "parent_asin": "i3",
            "title": "Game Controller",
            "main_category": "Video Games",
            "details": {"Type of item": "Accessory"},
        },
    ]
    review_rows = [
        {
            "user_id": f"u{i}",
            "parent_asin": "i1" if i < 5 else "i2",
            "rating": 5,
            "timestamp": i,
            "title": "Review",
            "text": f"Review {i}",
        }
        for i in range(1, 11)
    ] + [
        {
            "user_id": "hardware-user",
            "parent_asin": "i3",
            "rating": 5,
            "timestamp": 11,
            "title": "Review",
            "text": "Hardware review",
        }
    ]

    def fake_iter(url, chunk_bytes, provenance=None):
        rows = metadata_rows if "meta_" in url else review_rows
        encoded = json.dumps(rows, sort_keys=True).encode()
        if provenance is not None:
            provenance.update(
                {
                    "source_url": url,
                    "source_bytes": len(encoded),
                    "scan": "full_sequential_range_scan",
                    "chunk_bytes": chunk_bytes,
                    "chunk_count": 1,
                    "chunk_hashes": {"0": hashlib.sha256(encoded).hexdigest()},
                    "source_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        yield from rows

    monkeypatch.setattr(build_full_snapshot, "iter_jsonl_records", fake_iter)

    manifest = build_full_snapshot.build_full_snapshot(
        category="Video_Games",
        output_root=tmp_path / "processed",
        manifest_dir=tmp_path / "manifests",
        window_bytes=128,
        batch_size=3,
    )

    assert manifest["profile_type"] == "full-category-stream"
    assert manifest["benchmark_eligible"] is True
    assert manifest["scope_counts"]["reviews_scanned"] == 11
    assert manifest["row_counts"]["interactions"] == 10
    assert manifest["row_counts"]["item_metadata"] == 2
    assert manifest["row_counts"]["validation_targets"] >= 0
    assert Path(manifest["artifact_paths"]["test_targets"]).is_file()
