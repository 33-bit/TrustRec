"""Evidence-removal audits for explanation claims."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from trustrec.recommenders.contracts import Timestamp, stable_model_hash, timestamp_value

from .claims import (
    ClaimEvidence,
    Explanation,
    ExplanationClaim,
    _normalize_rows,
)
from .evidence import aggregate_evidence


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _claim_evidence(value: Mapping[str, Any]) -> ClaimEvidence:
    sentiment = value.get("sentiment", value.get("polarity", "neutral"))
    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "neutral"
    timestamp = value.get("source_timestamp", value.get("timestamp"))
    if timestamp in (None, ""):
        raise ValueError("claim evidence must contain source_timestamp")
    return ClaimEvidence(
        review_id=_id(value.get("review_id"), "review_id"),
        item_id=_id(value.get("item_id"), "item_id"),
        aspect=_id(value.get("aspect"), "aspect"),
        text=_id(value.get("text", value.get("evidence")), "text"),
        sentiment=sentiment,
        confidence=float(value.get("confidence", 1.0)),
        source_timestamp=timestamp,
        start_offset=value.get("start_offset", value.get("start")),
        end_offset=value.get("end_offset", value.get("end")),
        author_id=value.get("author_id", value.get("user_id")),
        duplicate_group=value.get("duplicate_group"),
        sentiment_score=float(value.get("sentiment_score", 0.0)),
    )


def _claim_from_input(
    value: Explanation | ExplanationClaim | Mapping[str, Any],
) -> tuple[str, ExplanationClaim | None]:
    if isinstance(value, ExplanationClaim):
        return value.status, value
    if isinstance(value, Explanation):
        return value.status, value.claims[0] if value.claims else None
    if not isinstance(value, Mapping):
        raise TypeError("explanation must be an Explanation or mapping")
    explanation_status = str(value.get("status", value.get("explanation_status", "unavailable")))
    claims = value.get("claims")
    raw_claim: Mapping[str, Any] | None = None
    if isinstance(claims, list) and claims:
        candidate = claims[0]
        if isinstance(candidate, Mapping):
            raw_claim = candidate
    elif "aspect" in value:
        raw_claim = value
    if raw_claim is None:
        return explanation_status, None
    evidence_values = raw_claim.get("evidence", ())
    refs = tuple(_claim_evidence(item) for item in evidence_values if isinstance(item, Mapping))
    return (
        str(raw_claim.get("status", explanation_status)),
        ExplanationClaim(
            item_id=_id(raw_claim.get("item_id"), "item_id"),
            aspect=_id(raw_claim.get("aspect"), "aspect"),
            claim=raw_claim.get("claim"),
            sentiment_direction=str(raw_claim.get("sentiment_direction", "neutral")),
            status=str(raw_claim.get("status", explanation_status)),
            evidence=refs,
            support=float(raw_claim.get("support", 0.0)),
            distinct_author_count=int(raw_claim.get("distinct_author_count", 0)),
            effective_sample_size=float(raw_claim.get("effective_sample_size", 0.0)),
            score_contribution=float(raw_claim.get("score_contribution", 0.0)),
            conflicting=bool(raw_claim.get("conflicting", False)),
            refusal_reason=raw_claim.get("refusal_reason"),
        ),
    )


def _as_rows(rows: Iterable[ClaimEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "review_id": row.review_id,
            "item_id": row.item_id,
            "aspect": row.aspect,
            "sentiment_score": row.sentiment_score,
            "confidence": row.confidence,
            "timestamp": row.source_timestamp,
            "author_id": row.author_id,
            "duplicate_group": row.duplicate_group,
            "text": row.text,
        }
        for row in rows
    ]


def _local_aspect_score(
    rows: list[ClaimEvidence],
    *,
    item_id: str,
    aspect: str,
    cutoff_timestamp: Timestamp,
    prior_scores: Mapping[Any, Any] | None,
    user_aspect_weights: Mapping[str, float] | None,
    shrinkage_lambda: float,
    recency_half_life_days: float,
    duplicate_penalty: str | float,
) -> float:
    profiles = aggregate_evidence(
        _as_rows(rows),
        cutoff_timestamp=cutoff_timestamp,
        prior_scores=prior_scores,
        shrinkage_lambda=shrinkage_lambda,
        recency_half_life_days=recency_half_life_days,
        duplicate_penalty=duplicate_penalty,
        items=[item_id],
        aspects=[aspect],
    )
    profile = profiles[(item_id, aspect)]
    score = profile.score
    prior = profile.prior_score
    if user_aspect_weights is not None:
        weight = float(user_aspect_weights.get(aspect, 0.0))
        return weight * (score - prior)
    return score


@dataclass(frozen=True)
class FaithfulnessAudit:
    """One deterministic removal audit with original and recomputed parts."""

    recommendation_id: str
    user_id: str
    item_id: str
    snapshot_id: str
    claim: str | None
    aspect: str | None
    sentiment_direction: str | None
    evidence_review_id: str | None
    evidence_review_ids: tuple[str, ...]
    evidence_offsets: tuple[dict[str, int], ...]
    support_status: str
    audit_result: str
    original_score: float
    recomputed_score: float
    score_delta: float
    random_recomputed_score: float | None
    random_score_delta: float | None
    original_contributions: dict[str, float]
    recomputed_contributions: dict[str, float]
    random_recomputed_contributions: dict[str, float]
    removed_review_ids: tuple[str, ...]
    random_removed_review_ids: tuple[str, ...]
    seed: int
    candidate_set_hash: str | None
    normalization_fixed: bool
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        for value, field in (
            (self.recommendation_id, "recommendation_id"),
            (self.user_id, "user_id"),
            (self.item_id, "item_id"),
            (self.snapshot_id, "snapshot_id"),
        ):
            _id(value, field)
        if self.support_status not in {"supported", "insufficient_support", "unavailable"}:
            raise ValueError("invalid support_status")
        if self.audit_result not in {"faithful", "not_faithful", "abstained", "unavailable"}:
            raise ValueError("invalid audit_result")
        for value, field in (
            (self.original_score, "original_score"),
            (self.recomputed_score, "recomputed_score"),
            (self.score_delta, "score_delta"),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")

    @property
    def original_component_contributions(self) -> dict[str, float]:
        return self.original_contributions

    @property
    def recomputed_component_contributions(self) -> dict[str, float]:
        return self.recomputed_contributions

    @property
    def audit_status(self) -> str:
        return self.audit_result

    @property
    def original_component_scores(self) -> dict[str, float]:
        return self.original_contributions

    @property
    def recomputed_component_scores(self) -> dict[str, float]:
        return self.recomputed_contributions

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "evidence_review_ids",
            "removed_review_ids",
            "random_removed_review_ids",
        ):
            result[field] = list(result[field])
        result["evidence_offsets"] = [dict(offset) for offset in self.evidence_offsets]
        return result


def _score_parts(parts: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(parts, Mapping) or not parts:
        raise ValueError("score_parts must be a non-empty mapping")
    result: dict[str, float] = {}
    for key, value in parts.items():
        _id(str(key), "score part")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("score parts must be numeric") from error
        if not math.isfinite(numeric):
            raise ValueError("score parts must be finite")
        result[str(key)] = numeric
    return result


def audit_faithfulness(
    recommendation_id: str,
    user_id: str,
    item_id: str,
    snapshot_id: str,
    explanation: Explanation | ExplanationClaim | Mapping[str, Any] | None = None,
    evidence_rows: Iterable[Mapping[str, Any] | Any] = (),
    cutoff_timestamp: Timestamp = 0,
    score_parts: Mapping[str, float] | None = None,
    user_aspect_weights: Mapping[str, float] | None = None,
    prior_scores: Mapping[Any, Any] | None = None,
    shrinkage_lambda: float = 5.0,
    recency_half_life_days: float = 0.0,
    duplicate_penalty: str | float = "inverse_group_size",
    candidate_item_ids: Iterable[str] | None = None,
    seed: int = 7,
    min_effect: float = 1e-12,
    claim: Explanation | ExplanationClaim | Mapping[str, Any] | None = None,
) -> FaithfulnessAudit:
    """Remove cited evidence and compare its score change with random removal."""

    _id(recommendation_id, "recommendation_id")
    _id(user_id, "user_id")
    _id(item_id, "item_id")
    _id(snapshot_id, "snapshot_id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not math.isfinite(float(min_effect)) or min_effect < 0:
        raise ValueError("min_effect must be finite and non-negative")
    if explanation is None:
        explanation = claim
    if explanation is None:
        explanation = {"status": "unavailable", "claims": []}
    parts = _score_parts(score_parts or {})
    original_score = sum(parts.values())
    support_status, selected_claim = _claim_from_input(explanation)
    candidate_hash = None
    if candidate_item_ids is not None:
        candidate_ids = tuple(sorted(_id(item, "candidate item_id") for item in candidate_item_ids))
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_item_ids must be unique")
        candidate_hash = stable_model_hash({"candidate_item_ids": candidate_ids})

    empty: tuple[str, ...] = ()
    if selected_claim is None:
        result = "abstained" if support_status == "insufficient_support" else "unavailable"
        return FaithfulnessAudit(
            recommendation_id=recommendation_id,
            user_id=user_id,
            item_id=item_id,
            snapshot_id=snapshot_id,
            claim=None,
            aspect=None,
            sentiment_direction=None,
            evidence_review_id=None,
            evidence_review_ids=empty,
            evidence_offsets=(),
            support_status=support_status,
            audit_result=result,
            original_score=original_score,
            recomputed_score=original_score,
            score_delta=0.0,
            random_recomputed_score=None,
            random_score_delta=None,
            original_contributions=parts,
            recomputed_contributions=dict(parts),
            random_recomputed_contributions=dict(parts),
            removed_review_ids=empty,
            random_removed_review_ids=empty,
            seed=seed,
            candidate_set_hash=candidate_hash,
            normalization_fixed=True,
            refusal_reason="no_claim",
        )

    refs = tuple(selected_claim.evidence)
    cited_ids = tuple(sorted({ref.review_id for ref in refs}))
    offsets = tuple(
        {"start": ref.start_offset, "end": ref.end_offset}
        for ref in refs
        if ref.start_offset is not None and ref.end_offset is not None
    )
    if support_status != "supported" or not cited_ids:
        result = "abstained" if support_status == "insufficient_support" else "unavailable"
        return FaithfulnessAudit(
            recommendation_id=recommendation_id,
            user_id=user_id,
            item_id=item_id,
            snapshot_id=snapshot_id,
            claim=selected_claim.claim,
            aspect=selected_claim.aspect,
            sentiment_direction=selected_claim.sentiment_direction,
            evidence_review_id=cited_ids[0] if cited_ids else None,
            evidence_review_ids=cited_ids,
            evidence_offsets=offsets,
            support_status=support_status,
            audit_result=result,
            original_score=original_score,
            recomputed_score=original_score,
            score_delta=0.0,
            random_recomputed_score=None,
            random_score_delta=None,
            original_contributions=parts,
            recomputed_contributions=dict(parts),
            random_recomputed_contributions=dict(parts),
            removed_review_ids=cited_ids,
            random_removed_review_ids=empty,
            seed=seed,
            candidate_set_hash=candidate_hash,
            normalization_fixed=True,
            refusal_reason=selected_claim.refusal_reason or "no_claimed_evidence",
        )

    normalized = _normalize_rows(evidence_rows)
    cutoff_value = timestamp_value(cutoff_timestamp)
    for row in normalized:
        if timestamp_value(row.source_timestamp) >= cutoff_value:
            raise ValueError("evidence timestamp must be before cutoff")
    target_rows = [
        row for row in normalized if row.item_id == item_id and row.aspect == selected_claim.aspect
    ]
    if not target_rows:
        raise ValueError("cited evidence does not occur in evidence_rows")
    all_ids = tuple(sorted({row.review_id for row in target_rows}))
    missing = set(cited_ids) - set(all_ids)
    if missing:
        raise ValueError("cited evidence does not occur in evidence_rows")
    rng = random.Random(seed)
    random_pool = [review_id for review_id in all_ids if review_id not in cited_ids]
    if len(random_pool) < len(cited_ids):
        random_pool = list(all_ids)
    random_ids = tuple(sorted(rng.sample(random_pool, len(cited_ids))))

    def recompute(removed: tuple[str, ...]) -> tuple[float, dict[str, float]]:
        retained = [row for row in target_rows if row.review_id not in set(removed)]
        original_local = _local_aspect_score(
            target_rows,
            item_id=item_id,
            aspect=selected_claim.aspect,
            cutoff_timestamp=cutoff_timestamp,
            prior_scores=prior_scores,
            user_aspect_weights=user_aspect_weights,
            shrinkage_lambda=shrinkage_lambda,
            recency_half_life_days=recency_half_life_days,
            duplicate_penalty=duplicate_penalty,
        )
        local = _local_aspect_score(
            retained,
            item_id=item_id,
            aspect=selected_claim.aspect,
            cutoff_timestamp=cutoff_timestamp,
            prior_scores=prior_scores,
            user_aspect_weights=user_aspect_weights,
            shrinkage_lambda=shrinkage_lambda,
            recency_half_life_days=recency_half_life_days,
            duplicate_penalty=duplicate_penalty,
        )
        recomputed_parts = dict(parts)
        aspect_key = "aspect" if "aspect" in recomputed_parts else "aspect_contribution"
        if aspect_key not in recomputed_parts:
            recomputed_parts[aspect_key] = original_local
        recomputed_parts[aspect_key] += local - original_local
        return sum(recomputed_parts.values()), recomputed_parts

    recomputed_score, recomputed_parts = recompute(cited_ids)
    random_score, random_parts = recompute(random_ids)
    score_delta = original_score - recomputed_score
    random_delta = original_score - random_score
    faithful = abs(score_delta) > min_effect and abs(score_delta) > abs(random_delta)
    return FaithfulnessAudit(
        recommendation_id=recommendation_id,
        user_id=user_id,
        item_id=item_id,
        snapshot_id=snapshot_id,
        claim=selected_claim.claim,
        aspect=selected_claim.aspect,
        sentiment_direction=selected_claim.sentiment_direction,
        evidence_review_id=cited_ids[0],
        evidence_review_ids=cited_ids,
        evidence_offsets=offsets,
        support_status=support_status,
        audit_result="faithful" if faithful else "not_faithful",
        original_score=original_score,
        recomputed_score=recomputed_score,
        score_delta=score_delta,
        random_recomputed_score=random_score,
        random_score_delta=random_delta,
        original_contributions=parts,
        recomputed_contributions=recomputed_parts,
        random_recomputed_contributions=random_parts,
        removed_review_ids=cited_ids,
        random_removed_review_ids=random_ids,
        seed=seed,
        candidate_set_hash=candidate_hash,
        normalization_fixed=True,
    )


run_faithfulness_audit = audit_faithfulness
evidence_removal_audit = audit_faithfulness
audit_evidence_removal = audit_faithfulness
audit_explanation = audit_faithfulness
run_evidence_removal_test = audit_faithfulness
EvidenceRemovalAudit = FaithfulnessAudit


__all__ = [
    "FaithfulnessAudit",
    "EvidenceRemovalAudit",
    "audit_faithfulness",
    "audit_evidence_removal",
    "audit_explanation",
    "evidence_removal_audit",
    "run_faithfulness_audit",
    "run_evidence_removal_test",
]
