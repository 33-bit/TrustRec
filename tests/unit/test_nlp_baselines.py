from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from trustrec.nlp.baselines import (
    DictionaryBaseline,
    evaluate_predictions,
    split_text_units,
    train_tfidf_svm,
)


def test_split_text_units_keeps_source_offsets() -> None:
    text = "The story is engaging, but the game keeps crashing."

    units = split_text_units(text, review_id="r1", unit_id_prefix="u")

    assert [(unit.start, unit.end, unit.text) for unit in units] == [
        (0, 21, "The story is engaging"),
        (27, 51, "the game keeps crashing."),
    ]
    assert all(text[unit.start : unit.end] == unit.text for unit in units)


def test_dictionary_baseline_emits_contrasting_aspect_labels() -> None:
    text = "The story is engaging, but the game keeps crashing."

    prediction = DictionaryBaseline().predict(text, review_id="r1", unit_id="u1")

    labels = {(label["aspect"], label["polarity"]) for label in prediction["labels"]}
    assert labels == {("story", "positive"), ("performance", "negative")}
    assert all(
        text[item["start"] : item["end"]] == item["evidence"] for item in prediction["labels"]
    )
    assert prediction["refusal_reason"] is None


def test_tfidf_svm_uses_pilot_labels_and_predicts_target_sentiment() -> None:
    training = [
        {
            "review_id": "r1",
            "unit_id": "u1",
            "text": "The story is engaging.",
            "labels": [
                {
                    "aspect": "story",
                    "polarity": "positive",
                    "start": 4,
                    "end": 9,
                    "evidence": "story",
                }
            ],
        },
        {
            "review_id": "r2",
            "unit_id": "u2",
            "text": "The story is boring.",
            "labels": [
                {
                    "aspect": "story",
                    "polarity": "negative",
                    "start": 4,
                    "end": 9,
                    "evidence": "story",
                }
            ],
        },
        {
            "review_id": "r3",
            "unit_id": "u3",
            "text": "The graphics are beautiful.",
            "labels": [
                {
                    "aspect": "graphics",
                    "polarity": "positive",
                    "start": 4,
                    "end": 12,
                    "evidence": "graphics",
                }
            ],
        },
        {
            "review_id": "r4",
            "unit_id": "u4",
            "text": "The graphics are ugly.",
            "labels": [
                {
                    "aspect": "graphics",
                    "polarity": "negative",
                    "start": 4,
                    "end": 12,
                    "evidence": "graphics",
                }
            ],
        },
    ]

    for row in training:
        row.update(_provenance(row["text"], row["review_id"]))
    expanded = []
    for row in training:
        for index in range(3):
            copied = deepcopy(row)
            copied["unit_id"] = f"{row['unit_id']}-{index}"
            copied["review_id"] = f"{row['review_id']}-{index}"
            copied["item_id"] = f"i-{row['review_id']}-{index}"
            expanded.append(copied)
    model = train_tfidf_svm(expanded, seed=7, sentiment_threshold=0.5)
    prediction = model.predict("The story is engaging.", review_id="r5", unit_id="u5")

    assert {item["aspect"] for item in prediction["labels"]} == {"story"}
    assert {item["polarity"] for item in prediction["labels"]} == {"positive"}


def _contrast_training() -> list[dict[str, object]]:
    story_rows: list[dict[str, object]] = []
    graphics_rows: list[dict[str, object]] = []
    for index in range(12):
        story = "engaging" if index % 2 == 0 else "boring"
        text = f"The story is {story}."
        row = {"review_id": f"story-{index}", "unit_id": f"story-u{index}", "text": text}
        row.update(_provenance(text, f"story-{index}"))
        row["item_id"] = f"story-i{index}"
        row["labels"] = [
            {
                "aspect": "story",
                "polarity": "positive" if story == "engaging" else "negative",
                "start": 0,
                "end": len(text),
                "evidence": text,
            }
        ]
        story_rows.append(row)
        graphics = "beautiful" if index % 3 == 0 else "ugly"
        text = f"The graphics are {graphics}."
        row = {"review_id": f"graphics-{index}", "unit_id": f"graphics-u{index}", "text": text}
        row.update(_provenance(text, f"graphics-{index}"))
        row["item_id"] = f"graphics-i{index}"
        row["labels"] = [
            {
                "aspect": "graphics",
                "polarity": "positive" if graphics == "beautiful" else "negative",
                "start": 0,
                "end": len(text),
                "evidence": text,
            }
        ]
        graphics_rows.append(row)
    return story_rows + graphics_rows


def test_svm_preserves_opposite_polarities_for_two_aspects() -> None:
    model = train_tfidf_svm(_contrast_training(), seed=7, sentiment_threshold=0.5)
    text = "The story is engaging, but the graphics are ugly."

    result = model.predict(text, review_id="new", unit_id="new")

    assert {(label["aspect"], label["polarity"]) for label in result["labels"]} == {
        ("story", "positive"),
        ("graphics", "negative"),
    }
    assert all(
        text[label["start"] : label["end"]] == label["evidence"] for label in result["labels"]
    )


def test_svm_preserves_both_polarities_of_one_aspect() -> None:
    model = train_tfidf_svm(_contrast_training(), seed=7, sentiment_threshold=0.5)
    result = model.predict("The story is engaging. The story is boring.")

    assert {(label["aspect"], label["polarity"]) for label in result["labels"]} == {
        ("story", "positive"),
        ("story", "negative"),
    }


def test_svm_abstains_when_calibration_lacks_independent_groups() -> None:
    rows = _contrast_training()[:2]
    for row in rows:
        row["duplicate_group"] = "same"
        row["item_id"] = "same"

    model = train_tfidf_svm(rows, seed=7)
    result = model.predict("The story is engaging.")

    assert result["labels"] == []
    assert model.training_provenance["calibration"]["story"]["supported"] is False


def test_model_hash_changes_when_training_labels_change() -> None:
    rows = _contrast_training()
    first = train_tfidf_svm(rows, seed=7)
    changed = deepcopy(rows)
    changed[0]["labels"][0]["polarity"] = "negative"
    second = train_tfidf_svm(changed, seed=7)

    assert first.model_hash != second.model_hash


def _provenance(text: str, review_id: str) -> dict[str, object]:
    return {
        "split_role": "development_pilot",
        "label_status": "llm_silver",
        "source_snapshot_id": "snapshot-1",
        "source_snapshot_dataset_hash": "a" * 64,
        "source_cutoff_timestamp": "1970-01-01T00:00:10Z",
        "timestamp": 1,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "item_id": "i1",
        "duplicate_group": review_id,
        "unit_start": 0,
        "unit_end": len(text),
        "unit_text": text,
        "out_of_scope": False,
        "needs_review": False,
        "usage_restrictions": {"allowed_uses": ["optional_nlp_training"], "prohibited_uses": []},
    }


def test_dictionary_baseline_preserves_negation() -> None:
    prediction = DictionaryBaseline().predict(
        "The story is not good, but the graphics are not bad.", review_id="r1", unit_id="u1"
    )

    assert {(item["aspect"], item["polarity"]) for item in prediction["labels"]} == {
        ("story", "negative"),
        ("graphics", "positive"),
    }


def test_dictionary_baseline_abstains_from_unclear_sentiment() -> None:
    prediction = DictionaryBaseline().predict("The story exists.", review_id="r1", unit_id="u1")

    assert prediction["labels"] == []
    assert prediction["refusal_reason"] == "no_explicit_sentiment"


@pytest.mark.parametrize("role", ["llm_pseudo_test", "recommendation_test"])
def test_tfidf_svm_rejects_held_out_training_roles(role: str) -> None:
    record = {"unit_id": "u1", "review_id": "r1", "text": "The story is great.", "labels": []}
    record.update(_provenance(record["text"], "r1"))
    record["split_role"] = role

    with pytest.raises(ValueError, match="development_pilot"):
        train_tfidf_svm([record], seed=7)


def test_tfidf_svm_rejects_cutoff_boundary() -> None:
    record = {"unit_id": "u1", "review_id": "r1", "text": "The story is great.", "labels": []}
    record.update(_provenance(record["text"], "r1"))
    record["timestamp"] = 10000

    with pytest.raises(ValueError, match="cutoff"):
        train_tfidf_svm([record], seed=7)


def test_evaluate_predictions_reports_consistency_and_errors() -> None:
    records = [
        {
            "review_id": "r1",
            "unit_id": "u1",
            "text": "The story is engaging.",
            "labels": [
                {
                    "aspect": "story",
                    "polarity": "positive",
                    "start": 4,
                    "end": 9,
                    "evidence": "story",
                }
            ],
            "out_of_scope": False,
            "needs_review": False,
        },
        {
            "review_id": "r2",
            "unit_id": "u2",
            "text": "The story is boring.",
            "labels": [
                {
                    "aspect": "story",
                    "polarity": "negative",
                    "start": 4,
                    "end": 9,
                    "evidence": "story",
                }
            ],
            "out_of_scope": False,
            "needs_review": False,
        },
    ]
    predictions = [
        {
            "model": "dictionary",
            "review_id": "r1",
            "unit_id": "u1",
            "text": records[0]["text"],
            "labels": [
                {
                    "aspect": "story",
                    "polarity": "positive",
                    "start": 0,
                    "end": 22,
                    "evidence": records[0]["text"],
                }
            ],
            "refusal_reason": None,
        },
        {
            "model": "dictionary",
            "review_id": "r2",
            "unit_id": "u2",
            "text": records[1]["text"],
            "labels": [],
            "refusal_reason": "no_aspect_match",
        },
    ]

    metrics = evaluate_predictions(records, predictions)

    assert metrics["eligible_units"] == 2
    assert metrics["aspect"]["micro_f1"] < 1.0
    assert metrics["sentiment_on_detected_aspects"]["coverage"] == 0.5
    assert metrics["evidence"]["valid_rate"] == 1.0
    assert len(metrics["error_samples"]) == 1
