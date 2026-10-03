from scripts.run_laya_labeling import label_record


class FakeRouter:
    def predict(self, text, questions, model):
        assert text == "The story is excellent."
        assert "has_story" in questions
        assert model == "laya"
        return {
            "answers": {
                "has_story": {"noul": 0.95, "answer_confidence": 0.9},
                "sentiment_story": {"choice": "positive", "answer_confidence": 0.87},
                "out_of_scope": {"noul": 0.01},
                "needs_adjudication": {"noul": 0.02},
            },
            "routing": {"model": "laya"},
        }


def test_label_record_preserves_source_span_and_marks_preliminary() -> None:
    result = label_record(
        FakeRouter(),
        "laya",
        {
            "unit_id": "u1",
            "source_annotation_id": "a1",
            "review_id": "r1",
            "item_id": "i1",
            "timestamp": 1,
            "source_text_sha256": "hash",
            "start": 4,
            "end": 27,
            "text": "The story is excellent.",
        },
        0.70,
    )

    assert result["labels"] == [
        {
            "aspect": "story",
            "polarity": "positive",
            "start": 4,
            "end": 27,
            "evidence": "The story is excellent.",
        }
    ]
    assert result["provenance"]["label_status"] == "ai_preliminary"
    assert result["provenance"]["annotator"] == "laya-local"
