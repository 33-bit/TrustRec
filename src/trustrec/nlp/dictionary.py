"""Deterministic aspect and sentiment rules for T2.3."""

from __future__ import annotations

import re
from typing import Any

from .records import ASPECTS
from .text import split_text_units

ASPECT_TERMS: dict[str, tuple[str, ...]] = {
    "gameplay": (
        "gameplay",
        "game play",
        "combat",
        "mechanics",
        "pacing",
        "difficulty",
        "playability",
        "fun",
        "enjoyable",
        "boring",
        "tedious",
    ),
    "story": ("story", "storyline", "plot", "narrative", "character", "dialogue"),
    "graphics": (
        "graphic",
        "graphics",
        "visual",
        "visuals",
        "art style",
        "animation",
        "resolution",
        "texture",
    ),
    "performance": (
        "performance",
        "bug",
        "bugs",
        "crash",
        "crashes",
        "crashing",
        "freeze",
        "freezes",
        "freezing",
        "lag",
        "loading",
        "glitch",
        "stutter",
        "stability",
    ),
    "controls": (
        "control",
        "controls",
        "input",
        "mapping",
        "joystick",
        "camera",
        "responsive",
        "handling",
        "handle",
    ),
    "multiplayer": (
        "multiplayer",
        "online play",
        "local play",
        "co-op",
        "coop",
        "matchmaking",
        "two-player",
        "two player",
    ),
    "content_replay": (
        "content",
        "length",
        "endgame",
        "replay",
        "replayable",
        "play time",
        "hours",
        "missions",
        "levels",
    ),
    "value": (
        "value",
        "price",
        "pricey",
        "expensive",
        "overpriced",
        "worth",
        "cost",
        "money",
        "purchase",
        "buy",
    ),
}

POSITIVE_TERMS = (
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "brilliant",
    "cool",
    "easy",
    "engaging",
    "enjoyable",
    "excellent",
    "fantastic",
    "fun",
    "good",
    "great",
    "impressive",
    "interesting",
    "like",
    "liked",
    "love",
    "loved",
    "perfect",
    "recommend",
    "responsive",
    "satisfying",
    "smooth",
    "solid",
    "wonderful",
    "worth",
)
NEGATIVE_TERMS = (
    "annoying",
    "bad",
    "boring",
    "broken",
    "crash",
    "crashes",
    "crashing",
    "dreadful",
    "disappointing",
    "disappointed",
    "expensive",
    "frustrating",
    "glitch",
    "glitches",
    "hated",
    "hate",
    "lag",
    "lacking",
    "poor",
    "repetitive",
    "redundant",
    "rough",
    "slow",
    "stutter",
    "tedious",
    "terrible",
    "ugly",
    "worse",
    "worst",
)
NEUTRAL_TERMS = ("average", "fine", "mixed", "okay", "ok", "nothing special", "no complaints")
NEGATIONS = {
    "cannot",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hardly",
    "isn't",
    "never",
    "no",
    "not",
    "nothing",
    "wasn't",
    "without",
    "won't",
}
OUT_OF_SCOPE_TERMS = (
    "headset",
    "headphone",
    "earbud",
    "shipping",
    "packaging",
    "seller",
    "console",
    "cable",
    "adapter",
    "lens",
)


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


_ASPECT_PATTERNS = {
    aspect: tuple(_term_pattern(term) for term in terms) for aspect, terms in ASPECT_TERMS.items()
}
_POSITIVE_PATTERNS = tuple(_term_pattern(term) for term in POSITIVE_TERMS)
_NEGATIVE_PATTERNS = tuple(_term_pattern(term) for term in NEGATIVE_TERMS)
_NEUTRAL_PATTERNS = tuple(_term_pattern(term) for term in NEUTRAL_TERMS)
_OUT_OF_SCOPE_PATTERNS = tuple(_term_pattern(term) for term in OUT_OF_SCOPE_TERMS)


def _negated(text: str, start: int) -> bool:
    prefix = text[:start].casefold()
    tokens = re.findall(r"[\w']+", prefix)[-4:]
    return any(token in NEGATIONS for token in tokens)


def _sentiment_score(text: str) -> tuple[int, int]:
    score = 0
    cues = 0
    for polarity, patterns, sign in (
        ("positive", _POSITIVE_PATTERNS, 1),
        ("negative", _NEGATIVE_PATTERNS, -1),
    ):
        del polarity
        for pattern in patterns:
            for match in pattern.finditer(text):
                cues += 1
                score += -sign if _negated(text, match.start()) else sign
    if any(pattern.search(text) for pattern in _NEUTRAL_PATTERNS):
        cues += 1
    return score, cues


def clause_sentiment(text: str) -> tuple[str | None, float, str]:
    """Return polarity, rule score, and score type for one clause."""

    score, cues = _sentiment_score(text)
    if score > 0:
        return "positive", min(1.0, score / max(cues, 1)), "rule_score"
    if score < 0:
        return "negative", min(1.0, abs(score) / max(cues, 1)), "rule_score"
    if cues:
        return "neutral", 0.0, "rule_score"
    return None, 0.0, "rule_score"


def aspect_mentions(
    text: str,
    terms: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Find configured aspect terms in ``text``."""

    patterns = _ASPECT_PATTERNS
    if terms is not None and terms is not ASPECT_TERMS:
        patterns = {
            aspect: tuple(_term_pattern(term) for term in values)
            for aspect, values in terms.items()
        }
    mentions: dict[str, list[tuple[int, int]]] = {}
    for aspect in ASPECTS:
        spans = [
            (match.start(), match.end())
            for pattern in patterns.get(aspect, ())
            for match in pattern.finditer(text)
        ]
        if spans:
            mentions[aspect] = sorted(spans)
    return mentions


def is_out_of_scope(text: str) -> bool:
    """Return whether strong hardware or merchant terms occur in a review."""

    return any(pattern.search(text) for pattern in _OUT_OF_SCOPE_PATTERNS)


class DictionaryBaseline:
    """Predict explicit aspect sentiment with fixed lexical rules."""

    name = "dictionary"

    def __init__(self, *, aspect_terms: dict[str, tuple[str, ...]] | None = None) -> None:
        self.aspect_terms = aspect_terms or ASPECT_TERMS

    def predict_target_sentiment(self, text: str, aspect: str) -> dict[str, Any]:
        """Predict sentiment on a supplied target aspect clause."""

        if aspect not in ASPECTS:
            raise ValueError(f"unsupported aspect: {aspect}")
        polarity, confidence, score_kind = clause_sentiment(text)
        return {
            "aspect": aspect,
            "polarity": polarity,
            "confidence": confidence,
            "confidence_kind": score_kind,
            "sentiment_score": {
                "positive": confidence,
                "negative": -confidence,
                "neutral": 0.0,
                None: 0.0,
            }[polarity],
            "refusal_reason": None if polarity is not None else "no_explicit_sentiment",
        }

    def predict(
        self,
        text: str,
        *,
        review_id: str = "",
        unit_id: str = "",
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "empty_text",
            }
        if is_out_of_scope(text):
            return {
                "model": self.name,
                "review_id": review_id,
                "unit_id": unit_id,
                "text": text,
                "labels": [],
                "refusal_reason": "out_of_scope",
            }

        labels: list[dict[str, Any]] = []
        saw_aspect = False
        for clause in split_text_units(text, review_id=review_id, unit_id_prefix=unit_id or "unit"):
            mentions = aspect_mentions(clause.text, self.aspect_terms)
            if not mentions:
                continue
            saw_aspect = True
            polarity, confidence, score_kind = clause_sentiment(clause.text)
            if polarity is None:
                continue
            for aspect in sorted(mentions):
                labels.append(
                    {
                        "aspect": aspect,
                        "polarity": polarity,
                        "start": clause.start,
                        "end": clause.end,
                        "evidence": clause.text,
                        "confidence": confidence,
                        "confidence_kind": score_kind,
                        "sentiment_score": {
                            "positive": confidence,
                            "negative": -confidence,
                            "neutral": 0.0,
                        }[polarity],
                    }
                )
        labels = _deduplicate_labels(labels)
        refusal_reason = None
        if not labels:
            refusal_reason = "no_explicit_sentiment" if saw_aspect else "no_aspect_match"
        return {
            "model": self.name,
            "review_id": review_id,
            "unit_id": unit_id,
            "text": text,
            "labels": labels,
            "refusal_reason": refusal_reason,
        }


def _deduplicate_labels(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for label in labels:
        key = (label["aspect"], label["polarity"], label["start"], label["end"])
        unique.setdefault(key, label)
    return [unique[key] for key in sorted(unique)]
