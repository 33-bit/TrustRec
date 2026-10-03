import pytest

from trustrec.nlp.laya_adapter import ASPECTS, build_questions, normalize_prediction


def test_build_questions_contains_one_presence_and_sentiment_decision_per_aspect() -> None:
    questions = build_questions()

    assert len(questions) == len(ASPECTS) * 2 + 2
    for aspect in ASPECTS:
        assert questions[f"has_{aspect}"]["type"] == "noul"
        assert questions[f"sentiment_{aspect}"]["type"] == "choice"
        assert set(questions[f"sentiment_{aspect}"]["criteria"]) == {
            "positive",
            "negative",
            "neutral",
        }
    assert questions["out_of_scope"]["type"] == "noul"
    assert questions["needs_adjudication"]["type"] == "noul"


def test_normalize_prediction_keeps_confidence_and_filters_absent_aspects() -> None:
    raw = {
        "answers": {
            "has_gameplay": {"noul": 0.93, "answer_confidence": 0.88},
            "sentiment_gameplay": {"choice": "positive", "answer_confidence": 0.82},
            "has_story": {"noul": 0.21, "answer_confidence": 0.91},
            "sentiment_story": {"choice": "negative", "answer_confidence": 0.90},
            "out_of_scope": {"noul": 0.04, "answer_confidence": 0.95},
            "needs_adjudication": {"noul": 0.12, "answer_confidence": 0.94},
        },
        "routing": {"model": "laya"},
    }
    for aspect in ASPECTS:
        raw["answers"].setdefault(f"has_{aspect}", {"noul": 0.01, "answer_confidence": 0.95})

    result = normalize_prediction("Review text", raw, min_presence=0.8)

    assert result["labels"] == [{"aspect": "gameplay", "polarity": "positive"}]
    assert result["confidence"]["has_gameplay"] == 0.88
    assert result["presence_probability"]["gameplay"] == 0.93
    assert result["out_of_scope"] is False
    assert result["needs_adjudication"] is False
    assert result["model"] == "laya"


def test_normalize_prediction_rejects_unknown_choice() -> None:
    raw = {"answers": {"has_gameplay": {"noul": 0.9}, "sentiment_gameplay": {"choice": "mixed"}}}

    with pytest.raises(ValueError, match="sentiment"):
        normalize_prediction("Review text", raw)


def test_low_answer_confidence_requires_review() -> None:
    raw = {
        "answers": {
            "has_gameplay": {"noul": 0.9, "answer_confidence": 0.95},
            "sentiment_gameplay": {"choice": "positive", "answer_confidence": 0.40},
            "out_of_scope": {"noul": 0.02, "answer_confidence": 0.95},
            "needs_adjudication": {"noul": 0.02, "answer_confidence": 0.95},
        }
    }
    result = normalize_prediction("Review text", raw)
    assert result["needs_adjudication"] is True
    assert "low_confidence:sentiment_gameplay" in result["review_reasons"]


def test_missing_answers_require_review_instead_of_silent_acceptance() -> None:
    result = normalize_prediction("Review text", {"answers": {}})
    assert result["needs_adjudication"] is True
    assert "missing:out_of_scope" in result["review_reasons"]
