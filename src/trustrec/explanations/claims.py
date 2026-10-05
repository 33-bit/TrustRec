"""Select traceable claims from snapshot-scoped aspect evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from trustrec.recommenders.contracts import Timestamp, timestamp_value

from .evidence import SENTIMENT_SCORES, AggregatedEvidence


def _required_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _timestamp(row: Mapping[str, Any]) -> Timestamp:
    value = row.get("timestamp", row.get("source_timestamp"))
    if value in (None, ""):
        raise ValueError("evidence row must contain timestamp or source_timestamp")
    return value


def _sentiment(row: Mapping[str, Any]) -> tuple[str, float]:
    value = row.get("sentiment", row.get("polarity"))
    if value in SENTIMENT_SCORES:
        return str(value), SENTIMENT_SCORES[str(value)]
    raw_score = row.get("sentiment_score", row.get("score"))
    if raw_score is None:
        raise ValueError("evidence row must contain sentiment or sentiment_score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as error:
        raise ValueError("sentiment_score must be numeric") from error
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("sentiment_score must be finite and between -1 and 1")
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", 0.0


def _expand_rows(rows: Iterable[Mapping[str, Any] | Any]) -> list[Mapping[str, Any]]:
    expanded: list[Mapping[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            source = dict(row)
        elif hasattr(row, "to_dict") and isinstance(row.to_dict(), Mapping):
            source = dict(row.to_dict())
        else:
            raise TypeError("evidence row must be a mapping")
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


def _offsets(source: Mapping[str, Any], evidence_text: str) -> tuple[int | None, int | None]:
    raw_start = source.get("start", source.get("evidence_start", source.get("start_offset")))
    raw_end = source.get("end", source.get("evidence_end", source.get("end_offset")))
    if raw_start is None and raw_end is None:
        offsets = source.get("evidence_offsets")
        if isinstance(offsets, list) and len(offsets) == 1 and isinstance(offsets[0], Mapping):
            raw_start = offsets[0].get("start")
            raw_end = offsets[0].get("end")
    full_text = source.get("source_text")
    if not isinstance(full_text, str):
        candidate_text = source.get("text")
        full_text = candidate_text if isinstance(candidate_text, str) else evidence_text
    if raw_start is None and raw_end is None:
        position = full_text.find(evidence_text)
        if position >= 0:
            raw_start, raw_end = position, position + len(evidence_text)
        else:
            return None, None
    if isinstance(raw_start, bool) or isinstance(raw_end, bool):
        raise ValueError("evidence offsets must be integers")
    if not isinstance(raw_start, int) or not isinstance(raw_end, int):
        raise ValueError("evidence offsets must be integers")
    if raw_start < 0 or raw_end <= raw_start or raw_end > len(full_text):
        raise ValueError("evidence offsets are outside source text")
    if full_text[raw_start:raw_end] != evidence_text:
        raise ValueError("evidence text does not match source text at offsets")
    return raw_start, raw_end


@dataclass(frozen=True)
class ClaimThresholds:
    """Validation thresholds for a strong explanation claim."""

    min_support: float = 0.25
    min_distinct_authors: int = 5
    min_effective_sample_size: float = 2.0
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.min_support)) or not 0.0 <= self.min_support <= 1.0:
            raise ValueError("min_support must be between 0 and 1")
        if (
            isinstance(self.min_distinct_authors, bool)
            or not isinstance(self.min_distinct_authors, int)
            or self.min_distinct_authors < 1
        ):
            raise ValueError("min_distinct_authors must be a positive integer")
        if (
            not math.isfinite(float(self.min_effective_sample_size))
            or self.min_effective_sample_size <= 0
        ):
            raise ValueError("min_effective_sample_size must be positive")
        if not math.isfinite(float(self.min_confidence)) or not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


@dataclass(frozen=True)
class ClaimEvidence:
    """One exact review passage that supports a claim."""

    review_id: str
    item_id: str
    aspect: str
    text: str
    sentiment: str
    confidence: float
    source_timestamp: Timestamp
    start_offset: int | None = None
    end_offset: int | None = None
    author_id: str | None = None
    duplicate_group: str | None = None
    sentiment_score: float = 0.0

    def __post_init__(self) -> None:
        for value, field in (
            (self.review_id, "review_id"),
            (self.item_id, "item_id"),
            (self.aspect, "aspect"),
            (self.text, "text"),
        ):
            _required_id(value, field)
        if self.sentiment not in SENTIMENT_SCORES:
            raise ValueError("sentiment must be positive, negative, or neutral")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        timestamp_value(self.source_timestamp)
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must be supplied together")
        if self.start_offset is not None and (
            self.start_offset < 0 or self.end_offset is None or self.end_offset <= self.start_offset
        ):
            raise ValueError("evidence offsets must describe a non-empty span")
        if self.author_id is not None:
            _required_id(self.author_id, "author_id")
        if self.duplicate_group is not None:
            _required_id(self.duplicate_group, "duplicate_group")

    @property
    def traceable(self) -> bool:
        return self.start_offset is not None and self.end_offset is not None

    @property
    def start(self) -> int | None:
        return self.start_offset

    @property
    def end(self) -> int | None:
        return self.end_offset

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if isinstance(self.source_timestamp, datetime):
            result["source_timestamp"] = self.source_timestamp.isoformat()
        result["start"] = self.start_offset
        result["end"] = self.end_offset
        return result


@dataclass(frozen=True)
class ExplanationClaim:
    """A claim with its support state and exact evidence references."""

    item_id: str
    aspect: str
    claim: str | None
    sentiment_direction: str
    status: str
    evidence: tuple[ClaimEvidence, ...]
    support: float
    distinct_author_count: int
    effective_sample_size: float
    score_contribution: float
    conflicting: bool
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        _required_id(self.item_id, "item_id")
        _required_id(self.aspect, "aspect")
        if self.sentiment_direction not in {"positive", "negative", "neutral", "conflicting"}:
            raise ValueError("invalid sentiment_direction")
        if self.status not in {"supported", "insufficient_support", "unavailable"}:
            raise ValueError("invalid claim status")
        if not math.isfinite(float(self.support)) or not 0.0 <= self.support <= 1.0:
            raise ValueError("support must be between 0 and 1")
        if self.distinct_author_count < 0 or self.effective_sample_size < 0:
            raise ValueError("support counts must be non-negative")
        if not math.isfinite(float(self.score_contribution)):
            raise ValueError("score_contribution must be finite")
        if self.status == "supported" and not self.claim:
            raise ValueError("supported claims must contain claim text")
        if self.status != "supported" and self.claim is not None:
            raise ValueError("refused claims cannot contain strong claim text")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = [item.to_dict() for item in self.evidence]
        result["refusal_message"] = self.refusal_message
        return result

    @property
    def evidence_refs(self) -> tuple[ClaimEvidence, ...]:
        return self.evidence

    @property
    def support_status(self) -> str:
        return self.status

    @property
    def sentiment(self) -> str:
        return self.sentiment_direction

    @property
    def refusal_message(self) -> str | None:
        if self.status == "supported":
            return None
        if self.refusal_reason:
            return f"Evidence for {self.aspect} is insufficient to support a strong claim."
        return f"Evidence for {self.aspect} is unavailable."


@dataclass(frozen=True)
class Explanation:
    """All claims and refusal reasons for one recommendation."""

    item_id: str
    user_id: str | None
    snapshot_id: str | None
    status: str
    claims: tuple[ExplanationClaim, ...]
    refusal_reasons: tuple[str, ...]
    score_parts: dict[str, float]
    dominant_score_part: str | None

    def __post_init__(self) -> None:
        _required_id(self.item_id, "item_id")
        if self.user_id is not None:
            _required_id(self.user_id, "user_id")
        if self.snapshot_id is not None:
            _required_id(self.snapshot_id, "snapshot_id")
        if self.status not in {"supported", "insufficient_support", "unavailable"}:
            raise ValueError("invalid explanation status")
        for key, value in self.score_parts.items():
            _required_id(key, "score part")
            if not math.isfinite(float(value)):
                raise ValueError("score parts must be finite")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["claims"] = [claim.to_dict() for claim in self.claims]
        result["dominant_score_text"] = self.dominant_score_text
        result["refusal_messages"] = [
            claim.refusal_message for claim in self.claims if claim.refusal_message
        ]
        return result

    @property
    def dominant_score_text(self) -> str | None:
        if self.dominant_score_part is None:
            return None
        return f"The {self.dominant_score_part} score part is the largest contribution."


def _profile_value(profile: AggregatedEvidence | Mapping[str, Any], name: str, default: Any) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(name, default)
    return getattr(profile, name, default)


def _normalize_rows(rows: Iterable[Mapping[str, Any] | Any]) -> list[ClaimEvidence]:
    normalized: list[ClaimEvidence] = []
    for source in _expand_rows(rows):
        review_id = _required_id(source.get("review_id"), "review_id")
        item_id = _required_id(source.get("item_id"), "item_id")
        aspect = _required_id(source.get("aspect"), "aspect")
        evidence_text = source.get("evidence", source.get("evidence_text", source.get("text")))
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            raise ValueError("evidence text must be non-empty")
        sentiment, sentiment_score = _sentiment(source)
        try:
            confidence = float(source.get("confidence", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("confidence must be numeric") from error
        author_id = source.get("author_id", source.get("user_id"))
        if author_id is not None:
            author_id = _required_id(author_id, "author_id")
        duplicate_group = source.get("duplicate_group")
        if duplicate_group is not None:
            duplicate_group = _required_id(duplicate_group, "duplicate_group")
        start, end = _offsets(source, evidence_text)
        normalized.append(
            ClaimEvidence(
                review_id=review_id,
                item_id=item_id,
                aspect=aspect,
                text=evidence_text,
                sentiment=sentiment,
                confidence=confidence,
                source_timestamp=_timestamp(source),
                start_offset=start,
                end_offset=end,
                author_id=author_id,
                duplicate_group=duplicate_group,
                sentiment_score=sentiment_score,
            )
        )
    return normalized


def _profile(profiles: Mapping[Any, Any], item_id: str, aspect: str) -> Any | None:
    result = profiles.get((item_id, aspect))
    if result is None:
        result = profiles.get(f"{item_id}:{aspect}")
    return result


def _dominant_score_part(score_parts: Mapping[str, float]) -> str | None:
    candidates = {
        key: float(value)
        for key, value in score_parts.items()
        if key not in {"gate", "support"}
        and not key.startswith("normalized_")
        and math.isfinite(float(value))
    }
    if not candidates:
        return None
    return max(candidates, key=lambda key: (abs(candidates[key]), key))


def _direction(profile: Any) -> tuple[str, bool]:
    positive = float(_profile_value(profile, "positive_mass", 0.0))
    negative = float(_profile_value(profile, "negative_mass", 0.0))
    conflicting = bool(_profile_value(profile, "conflicting", False)) or (
        positive > 0.0 and negative > 0.0
    )
    if conflicting:
        return "conflicting", True
    score = float(_profile_value(profile, "score", _profile_value(profile, "sentiment_score", 0.0)))
    prior = float(_profile_value(profile, "prior_score", 0.0))
    if score > prior:
        return "positive", False
    if score < prior:
        return "negative", False
    return "neutral", False


def _refusal_reasons(
    *,
    refs: tuple[ClaimEvidence, ...],
    support: float,
    author_count: int,
    effective_sample_size: float,
    average_confidence: float,
    thresholds: ClaimThresholds,
) -> list[str]:
    reasons: list[str] = []
    if not refs:
        reasons.append("no_evidence")
    if refs and not all(ref.traceable for ref in refs):
        reasons.append("evidence_offsets_missing")
    if support < thresholds.min_support:
        reasons.append("support_below_threshold")
    if author_count < thresholds.min_distinct_authors:
        reasons.append("distinct_authors_below_threshold")
    if effective_sample_size < thresholds.min_effective_sample_size:
        reasons.append("effective_sample_size_below_threshold")
    if average_confidence < thresholds.min_confidence:
        reasons.append("confidence_below_threshold")
    return reasons


def select_explanation(
    item_id: str,
    profiles: Mapping[Any, Any],
    evidence_rows: Iterable[Mapping[str, Any] | Any],
    user_aspect_weights: Mapping[str, float],
    cutoff_timestamp: Timestamp,
    user_id: str | None = None,
    snapshot_id: str | None = None,
    score_parts: Mapping[str, float] | None = None,
    thresholds: ClaimThresholds | None = None,
    max_claims: int = 1,
) -> Explanation:
    """Select ranked, traceable claims for one recommended item."""

    _required_id(item_id, "item_id")
    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be a mapping")
    if not isinstance(user_aspect_weights, Mapping):
        raise TypeError("user_aspect_weights must be a mapping")
    if isinstance(max_claims, bool) or not isinstance(max_claims, int) or max_claims < 1:
        raise ValueError("max_claims must be a positive integer")
    thresholds = thresholds or ClaimThresholds()
    cutoff_value = timestamp_value(cutoff_timestamp)
    normalized = _normalize_rows(evidence_rows)
    for row in normalized:
        if timestamp_value(row.source_timestamp) >= cutoff_value:
            raise ValueError("evidence timestamp must be before cutoff")

    grouped: dict[str, dict[str, ClaimEvidence]] = defaultdict(dict)
    for row in normalized:
        if row.item_id == item_id:
            current = grouped[row.aspect].get(row.review_id)
            if current is None or row.confidence > current.confidence:
                grouped[row.aspect][row.review_id] = row

    parts = {str(key): float(value) for key, value in (score_parts or {}).items()}
    candidates: list[tuple[float, str, Any, tuple[ClaimEvidence, ...]]] = []
    for aspect, raw_weight in user_aspect_weights.items():
        _required_id(str(aspect), "aspect")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as error:
            raise ValueError("user aspect weights must be numeric") from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("user aspect weights must be finite and non-negative")
        profile = _profile(profiles, item_id, str(aspect))
        if profile is None:
            continue
        prior = float(_profile_value(profile, "prior_score", 0.0))
        score = float(
            _profile_value(profile, "score", _profile_value(profile, "sentiment_score", 0.0))
        )
        refs = tuple(
            sorted(
                grouped.get(str(aspect), {}).values(),
                key=lambda row: (-row.confidence, row.review_id),
            )
        )
        candidates.append((abs(weight * (score - prior)), str(aspect), profile, refs))
    candidates.sort(key=lambda row: (-row[0], row[1]))

    claims: list[ExplanationClaim] = []
    refusal_reasons: list[str] = []
    for _, aspect, profile, refs in candidates[:max_claims]:
        support = float(_profile_value(profile, "support", 0.0))
        author_ids = {ref.author_id for ref in refs if ref.author_id is not None}
        author_count = len(author_ids)
        effective_sample_size = float(
            _profile_value(profile, "effective_sample_size", float(len(refs)))
        )
        average_confidence = sum(ref.confidence for ref in refs) / len(refs) if refs else 0.0
        reasons = _refusal_reasons(
            refs=refs,
            support=support,
            author_count=author_count,
            effective_sample_size=effective_sample_size,
            average_confidence=average_confidence,
            thresholds=thresholds,
        )
        direction, conflicting = _direction(profile)
        weight = float(user_aspect_weights[aspect])
        prior = float(_profile_value(profile, "prior_score", 0.0))
        score = float(
            _profile_value(profile, "score", _profile_value(profile, "sentiment_score", 0.0))
        )
        contribution = weight * (score - prior)
        status = "supported" if not reasons else "insufficient_support"
        claim_text: str | None = None
        if status == "supported":
            if direction == "conflicting":
                claim_text = f"Item {item_id} has mixed evidence for {aspect}."
            else:
                claim_text = f"Item {item_id} has {direction} evidence for {aspect}."
        else:
            refusal_reasons.extend(f"{aspect}:{reason}" for reason in reasons)
        claims.append(
            ExplanationClaim(
                item_id=item_id,
                aspect=aspect,
                claim=claim_text,
                sentiment_direction=direction,
                status=status,
                evidence=refs,
                support=support,
                distinct_author_count=author_count,
                effective_sample_size=effective_sample_size,
                score_contribution=contribution,
                conflicting=conflicting,
                refusal_reason=": ".join(reasons) if reasons else None,
            )
        )

    if not claims:
        status = "unavailable"
        refusal_reasons.append("no_aspect_profile")
    elif any(claim.status == "supported" for claim in claims):
        status = "supported"
    elif any(claim.status == "insufficient_support" for claim in claims):
        status = "insufficient_support"
    else:
        status = "unavailable"
    return Explanation(
        item_id=item_id,
        user_id=user_id,
        snapshot_id=snapshot_id,
        status=status,
        claims=tuple(claims),
        refusal_reasons=tuple(dict.fromkeys(refusal_reasons)),
        score_parts=parts,
        dominant_score_part=_dominant_score_part(parts),
    )


select_claims = select_explanation
build_explanation = select_explanation
select_explanation_claims = select_explanation
Claim = ExplanationClaim
EvidenceReference = ClaimEvidence


__all__ = [
    "ClaimEvidence",
    "Claim",
    "ClaimThresholds",
    "EvidenceReference",
    "Explanation",
    "ExplanationClaim",
    "build_explanation",
    "select_claims",
    "select_explanation",
    "select_explanation_claims",
]
