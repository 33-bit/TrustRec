from trustrec.evaluation.leakage import (
    LeakageError,
    assert_review_ids_disjoint,
    assert_timestamps_before,
)


def test_future_timestamp_is_rejected() -> None:
    rows = [{"review_id": "r1", "timestamp": 101}]
    try:
        assert_timestamps_before(rows, 101, "train")
    except LeakageError as error:
        assert "train" in str(error)
    else:
        raise AssertionError("expected future timestamp to fail")


def test_target_review_id_is_rejected_from_features() -> None:
    try:
        assert_review_ids_disjoint([{"review_id": "r1"}], [{"review_id": "r1"}], "validation")
    except LeakageError as error:
        assert "target review IDs" in str(error)
    else:
        raise AssertionError("expected overlapping target ID to fail")


def test_disjoint_feature_and_target_ids_pass() -> None:
    assert_review_ids_disjoint([{"review_id": "r1"}], [{"review_id": "r2"}], "validation")
