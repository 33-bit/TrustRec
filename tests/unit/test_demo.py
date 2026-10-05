"""Tests for the prepared-artifact Streamlit demo service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parents[2] / "app" / "demo_bundle.json"


def _demo_api():
    try:
        from app.demo import (
            DemoValidationError,
            load_demo_bundle,
            load_demo_bundle_with_fallback,
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"demo service is not implemented: {error}")
    return DemoValidationError, load_demo_bundle, load_demo_bundle_with_fallback


def test_demo_fixture_contains_the_five_minute_flow_records() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["snapshot_id"]
    assert payload["dataset_hash"]
    assert payload["cutoff_timestamp"]
    assert payload["configuration_hash"]
    assert payload["model_hashes"]
    assert len(payload["users"]) == 1
    assert payload["users"][0]["history_count"] <= 2
    assert set(payload["rankings"]) >= {"b0_most_popular", "t0_trustrec"}
    assert payload["evidence"]
    assert payload["metrics"]


def test_load_demo_bundle_returns_lineage_and_typed_records() -> None:
    _, load_demo_bundle, _ = _demo_api()

    bundle = load_demo_bundle(FIXTURE_PATH)

    assert bundle.snapshot_id == "video_games-full-d6c4efeb74aa"
    assert bundle.users["demo-user-low-history"].history_count == 2
    assert bundle.items["demo-item-a"].title == "Aperture Arcade"
    assert bundle.rankings["t0_trustrec"]["demo-user-low-history"]
    assert bundle.metrics[0].run_id == "t5-2-demo-run"


def test_load_demo_bundle_rejects_missing_dataset_hash(tmp_path: Path) -> None:
    DemoValidationError, load_demo_bundle, _ = _demo_api()
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload.pop("dataset_hash")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DemoValidationError, match="dataset_hash"):
        load_demo_bundle(path)


def test_load_demo_bundle_with_fallback_preserves_load_error(tmp_path: Path) -> None:
    _, _, load_demo_bundle_with_fallback = _demo_api()
    invalid_path = tmp_path / "missing.json"

    result = load_demo_bundle_with_fallback(invalid_path)

    assert result.used_fallback is True
    assert result.bundle.snapshot_id == "video_games-full-d6c4efeb74aa"
    assert result.load_error


def test_default_loader_marks_the_checked_in_bundle_as_synthetic(monkeypatch) -> None:
    _, _, load_demo_bundle_with_fallback = _demo_api()
    monkeypatch.delenv("TRUSTREC_DEMO_BUNDLE", raising=False)

    result = load_demo_bundle_with_fallback()

    assert result.used_fallback is True
    assert result.load_error == "no external demo bundle configured"


def _ranking_api():
    try:
        from app.demo import compare_rankings, load_demo_bundle, rank_recommendations
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"ranking service is not implemented: {error}")
    return compare_rankings, load_demo_bundle, rank_recommendations


def test_rank_recommendations_returns_stable_baseline_and_applies_k() -> None:
    _, load_demo_bundle, rank_recommendations = _ranking_api()
    bundle = load_demo_bundle(FIXTURE_PATH)

    result = rank_recommendations(
        bundle,
        user_id="demo-user-low-history",
        model_id="b0_most_popular",
        k=2,
    )

    assert [item.item_id for item in result] == ["demo-item-a", "demo-item-c"]
    assert [item.rank for item in result] == [1, 2]


def test_rank_recommendations_keeps_learned_weights_separate_from_priorities() -> None:
    _, load_demo_bundle, rank_recommendations = _ranking_api()
    bundle = load_demo_bundle(FIXTURE_PATH)
    defaults = rank_recommendations(
        bundle,
        user_id="demo-user-low-history",
        model_id="t0_trustrec",
        k=3,
    )
    graphics_priority = rank_recommendations(
        bundle,
        user_id="demo-user-low-history",
        model_id="t0_trustrec",
        k=3,
        priorities={"graphics": 1.5},
    )

    assert [item.item_id for item in defaults] == ["demo-item-a", "demo-item-c", "demo-item-b"]
    assert [item.item_id for item in graphics_priority] == [
        "demo-item-b",
        "demo-item-a",
        "demo-item-c",
    ]
    assert defaults[0].learned_aspect_weights == graphics_priority[1].learned_aspect_weights
    assert defaults[0].temporary_priorities == {}
    assert graphics_priority[0].temporary_priorities == {"graphics": 1.5}
    assert defaults[0].score_parts["mf"] == graphics_priority[1].score_parts["mf"]
    assert defaults[0].score_parts["graph"] == graphics_priority[1].score_parts["graph"]

    changed = _ranking_api()[0](defaults, graphics_priority)
    assert changed == ("demo-item-a", "demo-item-b", "demo-item-c")


def _evidence_api():
    try:
        from app.demo import evidence_for_item, evidence_state, load_demo_bundle
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"evidence service is not implemented: {error}")
    return evidence_for_item, evidence_state, load_demo_bundle


def test_evidence_states_cover_supported_conflicting_weak_and_unavailable() -> None:
    evidence_for_item, evidence_state, load_demo_bundle = _evidence_api()
    bundle = load_demo_bundle(FIXTURE_PATH)

    supported = evidence_for_item(bundle, "demo-item-a")
    conflicting = evidence_for_item(bundle, "demo-item-b")
    weak = evidence_for_item(bundle, "demo-item-c")

    assert evidence_state(supported) == "supported"
    assert evidence_state(conflicting) == "conflicting"
    assert evidence_state(weak) == "insufficient_support"
    assert evidence_state(()) == "unavailable"
    assert all("author_id" not in row.to_display() for row in conflicting)
    assert all("reviewer_id" not in row.to_display() for row in conflicting)


def test_evidence_for_item_uses_stable_source_order() -> None:
    evidence_for_item, _, load_demo_bundle = _evidence_api()
    bundle = load_demo_bundle(FIXTURE_PATH)

    rows = evidence_for_item(bundle, "demo-item-b")

    assert [row.review_id for row in rows] == ["demo-review-b1", "demo-review-b2"]
    assert rows[0].text == "The neon effects look sharp on a large screen."


def test_streamlit_view_exposes_main_entrypoint() -> None:
    try:
        from app.streamlit_app import main
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"Streamlit demo entrypoint is not implemented: {error}")

    assert callable(main)
