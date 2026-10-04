from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from scripts.build_annotation_pilot import build_pilot_rows


def test_pilot_filters_post_cutoff_rows(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    texts = [
        {"review_id": "r1", "item_id": "i1", "text": "Great game."},
        {"review_id": "r2", "item_id": "i2", "text": "Late game."},
    ]
    interactions = [
        {"review_id": "r1", "user_id": "u1", "item_id": "i1", "rating": 5.0, "timestamp": 1},
        {"review_id": "r2", "user_id": "u2", "item_id": "i2", "rating": 5.0, "timestamp": 2},
    ]
    parquet.write_table(pa.Table.from_pylist(texts), snapshot / "review_texts.parquet")
    parquet.write_table(pa.Table.from_pylist(interactions), snapshot / "interactions.parquet")
    with pytest.raises(ValueError, match="only"):
        build_pilot_rows(snapshot, "snapshot", "1970-01-01T00:00:00.000Z")
