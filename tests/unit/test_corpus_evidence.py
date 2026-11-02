import pytest

from trustrec.nlp.corpus import dictionary_fingerprint, label_reviews
from trustrec.nlp.dictionary import DictionaryBaseline
from trustrec.nlp.records import source_hash


def _review(text: str, **overrides: object) -> dict[str, object]:
    review = {
        "review_id": "r1",
        "item_id": "i1",
        "author_id": "u1",
        "timestamp": 5,
        "text": text,
        "source_text_sha256": source_hash(text),
        "duplicate_group": "g1",
        "rating": 1.0,
    }
    review.update(overrides)
    return review


def test_label_reviews_writes_offset_exact_evidence_rows() -> None:
    text = "The story is great. The controls are terrible."
    rows, summary = label_reviews([_review(text)], DictionaryBaseline(), label_source="dict")

    assert {(row["aspect"], row["sentiment"]) for row in rows} == {
        ("story", "positive"),
        ("controls", "negative"),
    }
    for row in rows:
        assert text[row["start"] : row["end"]] == row["evidence"]
        assert row["author_id"] == "u1"
        assert row["source_text_sha256"] == source_hash(text)
        assert "rating" not in row
    assert summary.labeled_reviews == 1
    assert summary.evidence_rows == 2


def test_label_reviews_ignores_star_ratings() -> None:
    text = "The graphics are beautiful."
    low, _ = label_reviews([_review(text, rating=1.0)], DictionaryBaseline(), label_source="d")
    high, _ = label_reviews([_review(text, rating=5.0)], DictionaryBaseline(), label_source="d")

    assert low == high


def test_label_reviews_counts_refusals() -> None:
    reviews = [
        _review("", review_id="empty"),
        _review("Arrived quickly in good packaging and the cable works.", review_id="scope"),
        _review("I bought this yesterday.", review_id="none"),
    ]
    rows, summary = label_reviews(reviews, DictionaryBaseline(), label_source="d")

    assert rows == []
    assert summary.reviews == 3
    assert sum(summary.refusals.values()) == 3


def test_label_reviews_rejects_changed_source_text() -> None:
    review = _review("The story is great.", source_text_sha256="0" * 64)

    with pytest.raises(ValueError, match="source_text_sha256"):
        label_reviews([review], DictionaryBaseline(), label_source="d")


def test_dictionary_fingerprint_is_stable() -> None:
    assert dictionary_fingerprint() == dictionary_fingerprint()
    assert len(dictionary_fingerprint()) == 64
