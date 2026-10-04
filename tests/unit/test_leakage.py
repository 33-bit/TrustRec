from trustrec.evaluation.leakage import (
    LeakageError,
    assert_metadata_before,
    assert_pseudo_test_not_used_for_tuning,
    assert_review_ids_disjoint,
    assert_same_evaluation_sets,
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


def test_missing_timestamp_is_rejected() -> None:
    try:
        assert_timestamps_before([{"review_id": "r1"}], 100, "features")
    except LeakageError as error:
        assert "timestamp" in str(error)
    else:
        raise AssertionError("expected missing timestamp to fail")


def test_missing_review_id_is_rejected() -> None:
    try:
        assert_review_ids_disjoint([{}], [{"review_id": "r2"}], "validation")
    except LeakageError as error:
        assert "review_id" in str(error)
    else:
        raise AssertionError("expected missing review ID to fail")


def test_compared_models_must_share_evaluation_sets() -> None:
    rows = {
        "user_id": [{"user_id": "u1"}],
        "candidate_set": [{"candidate_set": "c1"}],
        "target_set": [{"target_set": "t1"}],
    }
    other = {
        "user_id": [{"user_id": "u2"}],
        "candidate_set": [{"candidate_set": "c1"}],
        "target_set": [{"target_set": "t1"}],
    }
    try:
        assert_same_evaluation_sets({"a": rows, "b": other})
    except LeakageError as error:
        assert "user_id" in str(error)
    else:
        raise AssertionError("expected different user sets to fail")


def test_post_cutoff_metadata_is_rejected() -> None:
    try:
        assert_metadata_before([{"item_id": "i1", "metadata_timestamp": 101}], 101)
    except LeakageError as error:
        assert "post-cutoff" in str(error)
    else:
        raise AssertionError("expected post-cutoff metadata to fail")


def test_iso_timestamp_is_supported_for_evidence() -> None:
    assert_timestamps_before(
        [{"review_id": "r1", "timestamp": "2020-01-01T00:00:00Z"}],
        1577923200000,
        "evidence",
    )


def test_frozen_pseudo_test_tuning_is_rejected() -> None:
    try:
        assert_pseudo_test_not_used_for_tuning(
            {
                "split_role": "llm_pseudo_test",
                "frozen": True,
                "usage_restrictions": {"allowed_uses": ["prompt_tuning"]},
            }
        )
    except LeakageError as error:
        assert "tuning" in str(error)
    else:
        raise AssertionError("expected pseudo-test tuning to fail")
