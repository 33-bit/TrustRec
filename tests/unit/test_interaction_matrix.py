import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from trustrec.evaluation.protocol import build_evaluation_cases
from trustrec.recommenders.matrix import (
    build_interaction_matrix,
    candidate_pool_signature,
    top_k_order,
)


def _random_rows(seed: int = 3, count: int = 600) -> list[dict[str, object]]:
    generator = random.Random(seed)
    rows = []
    for position in range(count):
        rows.append(
            {
                "review_id": f"r{generator.randint(0, count // 2)}",
                "user_id": f"u{generator.randint(0, 59):02d}",
                "item_id": f"i{generator.randint(0, 24):02d}",
                "rating": float(generator.randint(1, 5)),
                "timestamp": generator.randint(0, 120),
            }
        )
        if position % 50 == 0:
            rows.append(dict(rows[-1], rating=1.0))
    return rows


def test_row_frame_and_arrow_paths_build_the_same_matrix() -> None:
    rows = _random_rows()
    from_rows = build_interaction_matrix(rows, 100)
    from_frame = build_interaction_matrix(pd.DataFrame(rows), 100)
    from_arrow = build_interaction_matrix(pa.Table.from_pylist(list(reversed(rows))), 100)

    assert from_rows.fingerprint == from_frame.fingerprint == from_arrow.fingerprint
    assert from_rows.user_ids == from_arrow.user_ids
    assert from_rows.item_ids == from_arrow.item_ids
    assert (from_rows.history != from_arrow.history).nnz == 0
    assert (from_rows.positive != from_arrow.positive).nnz == 0
    assert from_rows.popularity.tolist() == from_arrow.popularity.tolist()


def test_matrix_keeps_first_pre_cutoff_event_per_pair() -> None:
    rows = [
        {"review_id": "b", "user_id": "u", "item_id": "a", "rating": 5, "timestamp": 2},
        {"review_id": "a", "user_id": "u", "item_id": "a", "rating": 1, "timestamp": 2},
        {"review_id": "c", "user_id": "v", "item_id": "late", "rating": 5, "timestamp": 10},
    ]
    for source in (rows, pd.DataFrame(rows)):
        matrix = build_interaction_matrix(source, 10)

        assert matrix.item_ids == ("a",)
        assert matrix.positive.nnz == 0
        assert matrix.history.nnz == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("user_id", "", "user_id"),
        ("item_id", None, "item_id"),
        ("timestamp", True, "timestamp"),
        ("rating", float("inf"), "rating"),
    ],
)
def test_frame_path_rejects_invalid_rows(column: str, value: object, message: str) -> None:
    rows = [
        {"user_id": "u", "item_id": "a", "rating": 5.0, "timestamp": 1},
        {"user_id": "v", "item_id": "b", "rating": 4.0, "timestamp": 2},
    ]
    rows[1][column] = value
    frame = pd.DataFrame(rows)
    if column == "timestamp":
        frame["timestamp"] = [True, False]

    with pytest.raises(ValueError, match=message):
        build_interaction_matrix(frame, 10)


def test_candidate_signature_matches_the_evaluation_case() -> None:
    rows = _random_rows()
    targets = [
        {"review_id": "t1", "user_id": "u01", "item_id": "i03", "rating": 5.0, "timestamp": 130},
        {"review_id": "t2", "user_id": "u02", "item_id": "i04", "rating": 5.0, "timestamp": 130},
    ]
    dataset = build_evaluation_cases(rows, targets, snapshot_id="s", cutoff_timestamp=121)
    matrix = build_interaction_matrix(rows, 121)

    assert dataset.catalog_hash == matrix.catalog_hash
    for case in dataset.cases:
        assert case.candidate_signature == matrix.candidate_signature(case.user_id)
        assert case.candidate_count == matrix.candidate_indices(case.user_id).size
        assert case.candidate_signature == candidate_pool_signature(
            matrix.catalog_hash, case.history_item_ids
        )


def test_top_k_order_matches_a_full_sort_with_ties() -> None:
    generator = np.random.default_rng(5)
    for _ in range(50):
        primary = generator.integers(0, 4, size=40).astype(float)
        secondary = generator.integers(0, 3, size=40).astype(float)
        tie_break = generator.permutation(40)
        expected = sorted(range(40), key=lambda i: (-primary[i], -secondary[i], tie_break[i]))
        for k in (1, 7, 40, None):
            result = top_k_order(primary, tie_break, k, secondary=secondary)
            assert result.tolist() == expected[: (k or 40)]
