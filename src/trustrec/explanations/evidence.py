"""Snapshot-safe aggregation of aspect evidence.

The functions in this module accept mapping rows so they can consume Parquet,
JSON Lines, or pandas records without adding a data-frame dependency to the
explanation layer. They keep one contribution per review and aspect after
clause aggregation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from trustrec.recommenders.contracts import Timestamp, timestamp_value

DAY_MS = 86_400_000.0
SENTIMENT_SCORES = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _mapping(row: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "to_dict"):
        value = row.to_dict()
        if isinstance(value, Mapping):
            return value
    if hasattr(row, "__dataclass_fields__"):
        value = asdict(row)
        if isinstance(value, Mapping):
            return value
    raise TypeError("evidence row must be a mapping")


def _required_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sentiment_score(row: Mapping[str, Any]) -> float:
    value = row.get("sentiment_score", row.get("score"))
    if value is None:
        sentiment = row.get("sentiment", row.get("polarity"))
        if sentiment not in SENTIMENT_SCORES:
            raise ValueError("evidence sentiment must be positive, negative, or neutral")
        value = SENTIMENT_SCORES[sentiment]
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("sentiment_score must be numeric") from error
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("sentiment_score must be finite and between -1 and 1")
    return score


def _timestamp(row: Mapping[str, Any]) -> Timestamp:
    value = row.get("timestamp", row.get("source_timestamp"))
    if value in (None, ""):
        raise ValueError("evidence row must contain timestamp or source_timestamp")
    return value


@dataclass(frozen=True)
class EvidenceObservation:
    """One normalized aspect observation from a review."""

    review_id: str
    item_id: str
    aspect: str
    sentiment_score: float
    confidence: float
    timestamp: Timestamp
    author_id: str | None = None
    duplicate_group: str | None = None
    text: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any] | Any) -> EvidenceObservation:
        source = _mapping(row)
        review_id = _required_id(source.get("review_id"), "review_id")
        item_id = _required_id(source.get("item_id"), "item_id")
        aspect = _required_id(source.get("aspect"), "aspect")
        try:
            confidence = float(source.get("confidence", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("confidence must be numeric") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        author = source.get("author_id", source.get("user_id"))
        if author is not None:
            author = _required_id(author, "author_id")
        duplicate_group = source.get("duplicate_group")
        if duplicate_group is not None:
            duplicate_group = _required_id(duplicate_group, "duplicate_group")
        return cls(
            review_id=review_id,
            item_id=item_id,
            aspect=aspect,
            sentiment_score=_sentiment_score(source),
            confidence=confidence,
            timestamp=_timestamp(source),
            author_id=author,
            duplicate_group=duplicate_group,
            text=source.get("text", source.get("evidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EvidenceRow = EvidenceObservation


@dataclass(frozen=True)
class AggregatedEvidence:
    """An item and aspect profile with auditable support statistics."""

    item_id: str
    aspect: str
    score: float
    prior_score: float
    support: float
    total_weight: float
    positive_mass: float
    negative_mass: float
    neutral_mass: float
    author_count: int
    effective_sample_size: float
    review_count: int

    @property
    def conflicting(self) -> bool:
        return self.positive_mass > 0.0 and self.negative_mass > 0.0

    @property
    def sentiment_score(self) -> float:
        """Compatibility name for the shrunken aspect score."""

        return self.score

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["conflicting"] = self.conflicting
        result["sentiment_score"] = self.score
        return result


def _expand_rows(rows: Iterable[Mapping[str, Any] | Any]) -> list[Mapping[str, Any]]:
    """Expand annotation records that contain a nested ``labels`` list."""

    expanded: list[Mapping[str, Any]] = []
    for row in rows:
        source = _mapping(row)
        labels = source.get("labels")
        if isinstance(labels, list):
            for label in labels:
                if not isinstance(label, Mapping):
                    raise ValueError("labels must contain mapping objects")
                merged = dict(source)
                merged.update(label)
                expanded.append(merged)
        else:
            expanded.append(source)
    return expanded


def _normalize_rows(rows: Iterable[Mapping[str, Any] | Any]) -> list[EvidenceObservation]:
    return [EvidenceObservation.from_mapping(row) for row in _expand_rows(rows)]


def _duplicate_group_sizes(rows: Iterable[EvidenceObservation]) -> dict[tuple[str, str], int]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.duplicate_group is not None:
            groups[(row.item_id, row.duplicate_group)].add(row.review_id)
    return {key: len(review_ids) for key, review_ids in groups.items()}


def review_weight(
    row: Mapping[str, Any] | EvidenceObservation,
    cutoff_timestamp: Timestamp,
    *,
    recency_half_life_days: float = 0.0,
    duplicate_group_size: int | None = None,
    duplicate_penalty: str | float = "inverse_group_size",
) -> float:
    """Return the confidence, duplicate, and recency weight for one review."""

    source = row if isinstance(row, EvidenceObservation) else EvidenceObservation.from_mapping(row)
    cutoff = timestamp_value(cutoff_timestamp)
    source_timestamp = timestamp_value(source.timestamp)
    if source_timestamp >= cutoff:
        raise ValueError("evidence timestamp must be before cutoff")
    if not math.isfinite(float(recency_half_life_days)) or recency_half_life_days < 0:
        raise ValueError("recency_half_life_days must be finite and non-negative")
    if duplicate_group_size is None:
        source_mapping = row if isinstance(row, Mapping) else None
        duplicate_group_size = (
            source_mapping.get("duplicate_group_size", 1) if source_mapping is not None else 1
        )
    if (
        isinstance(duplicate_group_size, bool)
        or not isinstance(duplicate_group_size, int)
        or duplicate_group_size < 1
    ):
        raise ValueError("duplicate_group_size must be a positive integer")
    if duplicate_penalty == "inverse_group_size":
        duplicate_factor = 1.0 / int(duplicate_group_size)
    else:
        try:
            duplicate_factor = float(duplicate_penalty)
        except (TypeError, ValueError) as error:
            raise ValueError("duplicate_penalty must be inverse_group_size or numeric") from error
        if not math.isfinite(duplicate_factor) or duplicate_factor < 0:
            raise ValueError("duplicate_penalty must be finite and non-negative")
    if recency_half_life_days == 0:
        recency_factor = 1.0
    else:
        age_days = (cutoff - source_timestamp) / DAY_MS
        recency_factor = math.exp(-math.log(2.0) * age_days / recency_half_life_days)
    return source.confidence * duplicate_factor * recency_factor


def _review_contributions(
    rows: list[EvidenceObservation],
    cutoff_timestamp: Timestamp,
    *,
    recency_half_life_days: float,
    duplicate_penalty: str | float,
) -> dict[tuple[str, str, str], tuple[EvidenceObservation, float, float, float, float, float]]:
    group_sizes = _duplicate_group_sizes(rows)
    clauses: dict[tuple[str, str, str], list[tuple[EvidenceObservation, float]]] = defaultdict(list)
    for row in rows:
        group_size = group_sizes.get((row.item_id, row.duplicate_group or ""), 1)
        weight = review_weight(
            row,
            cutoff_timestamp,
            recency_half_life_days=recency_half_life_days,
            duplicate_group_size=group_size,
            duplicate_penalty=duplicate_penalty,
        )
        clauses[(row.item_id, row.aspect, row.review_id)].append((row, weight))

    contributions: dict[
        tuple[str, str, str], tuple[EvidenceObservation, float, float, float, float, float]
    ] = {}
    for key, values in clauses.items():
        total_clause_weight = sum(weight for _, weight in values)
        if total_clause_weight == 0:
            normalized = [(row, 0.0) for row, _ in values]
        else:
            normalized = [(row, weight / total_clause_weight) for row, weight in values]
        score = sum(row.sentiment_score * fraction for row, fraction in normalized)
        positive_fraction = sum(
            max(row.sentiment_score, 0.0) * fraction for row, fraction in normalized
        )
        negative_fraction = sum(
            max(-row.sentiment_score, 0.0) * fraction for row, fraction in normalized
        )
        neutral_fraction = max(0.0, 1.0 - positive_fraction - negative_fraction)
        # max keeps a review at one contribution after clause aggregation.
        review_weight_value = max(weight for _, weight in values)
        representative = values[0][0]
        contributions[key] = (
            representative,
            review_weight_value,
            score,
            positive_fraction,
            negative_fraction,
            neutral_fraction,
        )
    return contributions


def _prior_for(prior_scores: Mapping[Any, Any] | None, item_id: str, aspect: str) -> float:
    if not prior_scores:
        return 0.0
    value = prior_scores.get((item_id, aspect), prior_scores.get(aspect, 0.0))
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("prior scores must be numeric") from error
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("prior scores must be finite and between -1 and 1")
    return value


def aggregate_evidence(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    cutoff_timestamp: Timestamp,
    prior_scores: Mapping[Any, Any] | None = None,
    shrinkage_lambda: float = 5.0,
    recency_half_life_days: float = 0.0,
    duplicate_penalty: str | float = "inverse_group_size",
    items: Iterable[str] | None = None,
    aspects: Iterable[str] | None = None,
) -> dict[tuple[str, str], AggregatedEvidence]:
    """Aggregate evidence into item-aspect profiles before a cutoff."""

    if not math.isfinite(float(shrinkage_lambda)) or shrinkage_lambda <= 0:
        raise ValueError("shrinkage_lambda must be finite and positive")
    normalized = _normalize_rows(rows)
    contributions = _review_contributions(
        normalized,
        cutoff_timestamp,
        recency_half_life_days=recency_half_life_days,
        duplicate_penalty=duplicate_penalty,
    )
    item_values = {row.item_id for row in normalized}
    aspect_values = {row.aspect for row in normalized}
    if items is not None:
        item_values.update(_required_id(item_id, "item_id") for item_id in items)
    if aspects is not None:
        aspect_values.update(_required_id(aspect, "aspect") for aspect in aspects)
    if prior_scores:
        aspect_values.update(key for key in prior_scores if isinstance(key, str))
        item_values.update(
            key[0] for key in prior_scores if isinstance(key, tuple) and len(key) == 2
        )

    output: dict[tuple[str, str], AggregatedEvidence] = {}
    grouped: dict[
        tuple[str, str], list[tuple[EvidenceObservation, float, float, float, float, float]]
    ] = defaultdict(list)
    for (item_id, aspect, _), value in contributions.items():
        grouped[(item_id, aspect)].append(value)

    for item_id in sorted(item_values):
        for aspect in sorted(aspect_values):
            evidence = grouped.get((item_id, aspect), [])
            prior = _prior_for(prior_scores, item_id, aspect)
            total_weight = sum(value[1] for value in evidence)
            weighted_score = sum(value[1] * value[2] for value in evidence)
            score = (shrinkage_lambda * prior + weighted_score) / (shrinkage_lambda + total_weight)
            support = total_weight / (shrinkage_lambda + total_weight)
            positive_mass = sum(value[1] * value[3] for value in evidence)
            negative_mass = sum(value[1] * value[4] for value in evidence)
            neutral_mass = sum(value[1] * value[5] for value in evidence)
            squared_weight = sum(value[1] ** 2 for value in evidence)
            effective_sample_size = total_weight**2 / squared_weight if squared_weight else 0.0
            authors = {value[0].author_id for value in evidence if value[0].author_id is not None}
            output[(item_id, aspect)] = AggregatedEvidence(
                item_id=item_id,
                aspect=aspect,
                score=score,
                prior_score=prior,
                support=support,
                total_weight=total_weight,
                positive_mass=positive_mass,
                negative_mass=negative_mass,
                neutral_mass=neutral_mass,
                author_count=len(authors),
                effective_sample_size=effective_sample_size,
                review_count=len(evidence),
            )
    return output


aggregate_item_aspects = aggregate_evidence
aggregate_aspect_evidence = aggregate_evidence


def compute_aspect_priors(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    cutoff_timestamp: Timestamp,
    aspects: Iterable[str] | None = None,
    recency_half_life_days: float = 0.0,
    duplicate_penalty: str | float = "inverse_group_size",
) -> dict[str, float]:
    """Compute sentiment priors from evidence before the supplied cutoff."""

    normalized = _normalize_rows(rows)
    contributions = _review_contributions(
        normalized,
        cutoff_timestamp,
        recency_half_life_days=recency_half_life_days,
        duplicate_penalty=duplicate_penalty,
    )
    totals: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)
    for (_, aspect, _), value in contributions.items():
        totals[aspect] += value[1] * value[2]
        weights[aspect] += value[1]
    aspect_values = set(aspects or ()) | set(totals)
    for aspect in aspect_values:
        _required_id(aspect, "aspect")
    return {
        aspect: (totals[aspect] / weights[aspect] if weights[aspect] else 0.0)
        for aspect in sorted(aspect_values)
    }


def build_user_aspect_weights(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    user_id: str,
    aspects: Iterable[str],
    alpha: float = 1.0,
    prior_distribution: Mapping[str, float] | None = None,
    cutoff_timestamp: Timestamp | None = None,
) -> dict[str, float]:
    """Build smoothed aspect priorities from one user's pre-cutoff reviews."""

    _required_id(user_id, "user_id")
    aspect_values = tuple(dict.fromkeys(_required_id(aspect, "aspect") for aspect in aspects))
    if not aspect_values:
        raise ValueError("aspects must contain at least one value")
    if not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if prior_distribution is None:
        prior = {aspect: 1.0 / len(aspect_values) for aspect in aspect_values}
    else:
        prior = {aspect: float(prior_distribution.get(aspect, 0.0)) for aspect in aspect_values}
        if any(not math.isfinite(value) or value < 0 for value in prior.values()):
            raise ValueError("prior_distribution values must be finite and non-negative")
        total_prior = sum(prior.values())
        if total_prior <= 0:
            raise ValueError("prior_distribution must have positive mass")
        prior = {aspect: value / total_prior for aspect, value in prior.items()}
    seen: set[tuple[str, str]] = set()
    for source in _expand_rows(rows):
        candidate_user = source.get("user_id", source.get("author_id"))
        if candidate_user != user_id:
            continue
        if cutoff_timestamp is not None and timestamp_value(_timestamp(source)) >= timestamp_value(
            cutoff_timestamp
        ):
            raise ValueError("evidence timestamp must be before cutoff")
        review_id = _required_id(source.get("review_id"), "review_id")
        aspect = source.get("aspect")
        if aspect in aspect_values:
            seen.add((review_id, aspect))
    counts = {
        aspect: sum(1 for _, seen_aspect in seen if seen_aspect == aspect)
        for aspect in aspect_values
    }
    denominator = sum(counts.values()) + alpha
    return {
        aspect: (counts[aspect] + alpha * prior[aspect]) / denominator for aspect in aspect_values
    }


user_aspect_weights = build_user_aspect_weights
compute_user_aspect_weights = build_user_aspect_weights


def personalized_aspect_score(
    user_weights: Mapping[str, float],
    item_profiles: Mapping[tuple[str, str], AggregatedEvidence] | Mapping[str, AggregatedEvidence],
    *,
    item_id: str,
    prior_scores: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    """Return the aspect deviation score and weighted evidence support."""

    score = 0.0
    support = 0.0
    for aspect, weight in user_weights.items():
        profile = item_profiles.get((item_id, aspect))
        if profile is None:
            profile = item_profiles.get(f"{item_id}:{aspect}")
        if profile is None:
            value = _prior_for(prior_scores, item_id, aspect)
            profile_support = 0.0
            prior = value
        else:
            value = float(getattr(profile, "score", profile))
            profile_support = float(getattr(profile, "support", 0.0))
            if prior_scores is None or aspect not in prior_scores:
                prior = float(getattr(profile, "prior_score", 0.0))
            else:
                prior = _prior_for(prior_scores, item_id, aspect)
        score += float(weight) * (value - prior)
        support += float(weight) * profile_support
    return score, support


aspect_match_score = personalized_aspect_score
