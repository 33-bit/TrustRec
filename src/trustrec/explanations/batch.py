"""Explain many recommendations from per-item claim templates.

``select_explanation`` judges each (item, aspect) claim from that item's
evidence alone; only the choice of aspects and the score contribution depend on
the user. This module computes every (item, aspect) claim once and then builds
each user's explanation from the templates. ``explain_with_templates`` returns
the same statuses, aspects, directions, and contributions as
``select_explanation`` for the same inputs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .claims import ClaimThresholds, select_explanation
from .evidence import AggregatedEvidence

DISPLAY_EVIDENCE = 3


@dataclass(frozen=True)
class ClaimTemplate:
    """User-independent facts about one (item, aspect) claim."""

    item_id: str
    aspect: str
    status: str
    sentiment_direction: str
    conflicting: bool
    support: float
    distinct_author_count: int
    effective_sample_size: float
    average_confidence: float
    all_traceable: bool
    deviation: float
    evidence_count: int
    refusal_reason: str | None
    cited_evidence: tuple[dict[str, Any], ...] = field(default=())
    positive_share: float | None = None

    def status_under(self, thresholds: ClaimThresholds) -> str:
        """Return the claim status the same evidence gets under other thresholds."""

        reasons = (
            self.evidence_count == 0
            or not self.all_traceable
            or self.support < thresholds.min_support
            or self.distinct_author_count < thresholds.min_distinct_authors
            or self.effective_sample_size < thresholds.min_effective_sample_size
            or self.average_confidence < thresholds.min_confidence
        )
        return "insufficient_support" if reasons else "supported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "aspect": self.aspect,
            "status": self.status,
            "sentiment_direction": self.sentiment_direction,
            "conflicting": self.conflicting,
            "support": self.support,
            "distinct_author_count": self.distinct_author_count,
            "effective_sample_size": self.effective_sample_size,
            "average_confidence": self.average_confidence,
            "all_traceable": self.all_traceable,
            "deviation": self.deviation,
            "evidence_count": self.evidence_count,
            "refusal_reason": self.refusal_reason,
            "positive_share": self.positive_share,
            "cited_evidence": list(self.cited_evidence),
        }


def _cited(evidence: Sequence[Any], direction: str) -> tuple[dict[str, Any], ...]:
    """Pick up to three passages to show.

    A positive or negative claim shows two passages that match it and, when
    one exists, one passage from the other side (ADR-0010, ADR-0019). A mixed
    claim shows positive and negative passages. Ties keep the
    ``select_explanation`` order (confidence, then review ID).
    """

    positive = [ref for ref in evidence if ref.sentiment == "positive"]
    negative = [ref for ref in evidence if ref.sentiment == "negative"]
    if direction == "conflicting":
        # A mixed claim leads with whichever side has more passages.
        direction = "negative" if len(negative) > len(positive) else "positive"
    if direction == "negative":
        picked: list[Any] = negative[:2] + positive[:1]
    elif direction == "positive":
        picked = positive[:2] + negative[:1]
    else:
        picked = []
    chosen = {id(ref) for ref in picked}
    picked += [ref for ref in evidence if id(ref) not in chosen][: DISPLAY_EVIDENCE - len(picked)]
    return tuple(
        {
            "review_id": ref.review_id,
            "sentiment": ref.sentiment,
            "start": ref.start_offset,
            "end": ref.end_offset,
            "text": ref.text,
        }
        for ref in picked
    )


def build_claim_templates(
    item_id: str,
    profiles: Mapping[tuple[str, str], AggregatedEvidence],
    evidence_rows: Iterable[Mapping[str, Any]],
    aspects: Sequence[str],
    cutoff_timestamp: Any,
    thresholds: ClaimThresholds,
) -> dict[str, ClaimTemplate]:
    """Judge every aspect claim of one item once, from that item's evidence."""

    explanation = select_explanation(
        item_id,
        profiles,
        list(evidence_rows),
        {aspect: 1.0 for aspect in aspects},
        cutoff_timestamp,
        thresholds=thresholds,
        max_claims=len(aspects),
    )
    templates: dict[str, ClaimTemplate] = {}
    for claim in explanation.claims:
        refs = claim.evidence
        templates[claim.aspect] = ClaimTemplate(
            item_id=item_id,
            aspect=claim.aspect,
            status=claim.status,
            sentiment_direction=claim.sentiment_direction,
            conflicting=claim.conflicting,
            support=claim.support,
            distinct_author_count=claim.distinct_author_count,
            effective_sample_size=claim.effective_sample_size,
            average_confidence=sum(ref.confidence for ref in refs) / len(refs) if refs else 0.0,
            all_traceable=all(ref.traceable for ref in refs),
            deviation=claim.score_contribution,
            evidence_count=len(refs),
            refusal_reason=claim.refusal_reason,
            cited_evidence=_cited(refs, claim.sentiment_direction),
            positive_share=claim.positive_share,
        )
    return templates


def explain_with_templates(
    templates: Mapping[str, ClaimTemplate],
    user_aspect_weights: Mapping[str, float],
    *,
    max_claims: int = 1,
    aspect_in_score: bool = True,
) -> dict[str, Any]:
    """Build one recommendation's explanation from item templates and user weights.

    ``aspect_in_score`` says whether the aspect term is part of this item's
    ranking score. When it is not (for example, a cold-start fallback), the
    claims still describe the item, but each contribution is reported as 0.
    """

    candidates = []
    for aspect, raw_weight in user_aspect_weights.items():
        template = templates.get(aspect)
        if template is None:
            continue
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("user aspect weights must be finite and non-negative")
        candidates.append((abs(weight * template.deviation), aspect, weight, template))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    claims = []
    for _, aspect, weight, template in candidates[:max_claims]:
        claims.append(
            {
                "aspect": aspect,
                "status": template.status,
                "sentiment_direction": template.sentiment_direction,
                "positive_share": template.positive_share,
                "score_contribution": weight * template.deviation if aspect_in_score else 0.0,
                "refusal_reason": template.refusal_reason,
            }
        )
    statuses = {claim["status"] for claim in claims}
    if not claims:
        status = "unavailable"
    elif "supported" in statuses:
        status = "supported"
    else:
        status = "insufficient_support"
    return {"status": status, "claims": claims, "aspect_in_score": aspect_in_score}
