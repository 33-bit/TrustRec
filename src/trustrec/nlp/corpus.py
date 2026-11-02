"""Turn pre-cutoff review texts into flat aspect evidence rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import dictionary, text
from .records import source_hash

EVIDENCE_SCHEMA_VERSION = 1
DICTIONARY_LABELER_VERSION = "dictionary-evidence-v1"


class ReviewLabeler(Protocol):
    """A labeler returns aspect labels with offsets into the original text."""

    name: str

    def predict(self, text: str, *, review_id: str = "", unit_id: str = "") -> dict[str, Any]: ...


def dictionary_fingerprint() -> str:
    """Hash every rule that changes dictionary labels."""

    payload = {
        "version": DICTIONARY_LABELER_VERSION,
        "aspect_terms": {key: list(value) for key, value in dictionary.ASPECT_TERMS.items()},
        "positive_terms": list(dictionary.POSITIVE_TERMS),
        "negative_terms": list(dictionary.NEGATIVE_TERMS),
        "neutral_terms": list(dictionary.NEUTRAL_TERMS),
        "negations": sorted(dictionary.NEGATIONS),
        "out_of_scope_terms": list(dictionary.OUT_OF_SCOPE_TERMS),
        "boundary": text._BOUNDARY.pattern,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class LabelingSummary:
    """Counts that describe one labeling pass."""

    reviews: int = 0
    labeled_reviews: int = 0
    evidence_rows: int = 0
    refusals: Counter[str] = field(default_factory=Counter)
    aspects: Counter[str] = field(default_factory=Counter)
    sentiments: Counter[str] = field(default_factory=Counter)

    def update(self, other: LabelingSummary) -> None:
        self.reviews += other.reviews
        self.labeled_reviews += other.labeled_reviews
        self.evidence_rows += other.evidence_rows
        self.refusals.update(other.refusals)
        self.aspects.update(other.aspects)
        self.sentiments.update(other.sentiments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviews": self.reviews,
            "labeled_reviews": self.labeled_reviews,
            "evidence_rows": self.evidence_rows,
            "refusals": dict(sorted(self.refusals.items())),
            "aspects": dict(sorted(self.aspects.items())),
            "sentiments": dict(sorted(self.sentiments.items())),
        }


def _required(review: Mapping[str, Any], name: str) -> Any:
    value = review.get(name)
    if value is None or value == "":
        raise ValueError(f"review row is missing {name}")
    return value


def label_reviews(
    reviews: Iterable[Mapping[str, Any]],
    labeler: ReviewLabeler,
    *,
    label_source: str,
) -> tuple[list[dict[str, Any]], LabelingSummary]:
    """Label reviews and return one evidence row per aspect label.

    Each review needs ``review_id``, ``item_id``, ``author_id``, ``timestamp``,
    and ``text``. When ``source_text_sha256`` is present it must match the text.
    Evidence offsets refer to the original review text, and the evidence text
    always equals that slice. Ratings are never read.
    """

    rows: list[dict[str, Any]] = []
    summary = LabelingSummary()
    for review in reviews:
        review_id = _required(review, "review_id")
        item_id = _required(review, "item_id")
        author_id = _required(review, "author_id")
        timestamp = _required(review, "timestamp")
        review_text = review.get("text") or ""
        expected_hash = review.get("source_text_sha256")
        actual_hash = source_hash(review_text)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"review {review_id} text does not match source_text_sha256")
        summary.reviews += 1
        prediction = labeler.predict(review_text, review_id=review_id, unit_id=review_id)
        labels = prediction.get("labels", [])
        if not labels:
            summary.refusals[prediction.get("refusal_reason") or "no_label"] += 1
            continue
        summary.labeled_reviews += 1
        for label in labels:
            start, end = int(label["start"]), int(label["end"])
            if review_text[start:end] != label["evidence"]:
                raise ValueError(f"review {review_id} evidence does not match its offsets")
            rows.append(
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "review_id": review_id,
                    "item_id": item_id,
                    "author_id": author_id,
                    "timestamp": timestamp,
                    "duplicate_group": review.get("duplicate_group") or None,
                    "aspect": label["aspect"],
                    "sentiment": label["polarity"],
                    "sentiment_score": float(label["sentiment_score"]),
                    "confidence": float(label["confidence"]),
                    "confidence_kind": label.get("confidence_kind"),
                    "start": start,
                    "end": end,
                    "evidence": label["evidence"],
                    "source_text_sha256": actual_hash,
                    "label_source": label_source,
                }
            )
            summary.evidence_rows += 1
            summary.aspects[label["aspect"]] += 1
            summary.sentiments[label["polarity"]] += 1
    return rows, summary
