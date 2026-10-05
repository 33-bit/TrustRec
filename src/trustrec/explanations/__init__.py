"""Evidence selection and explanation rendering."""

from .aggregation import EvidenceAggregator
from .evidence import (
    AggregatedEvidence,
    EvidenceObservation,
    EvidenceRow,
    aggregate_aspect_evidence,
    aggregate_evidence,
    aggregate_item_aspects,
    aspect_match_score,
    build_user_aspect_weights,
    compute_aspect_priors,
    compute_user_aspect_weights,
    personalized_aspect_score,
    review_weight,
    user_aspect_weights,
)

__all__ = [
    "AggregatedEvidence",
    "EvidenceObservation",
    "EvidenceRow",
    "EvidenceAggregator",
    "aggregate_aspect_evidence",
    "aggregate_evidence",
    "aggregate_item_aspects",
    "aspect_match_score",
    "build_user_aspect_weights",
    "compute_aspect_priors",
    "compute_user_aspect_weights",
    "personalized_aspect_score",
    "review_weight",
    "user_aspect_weights",
]
