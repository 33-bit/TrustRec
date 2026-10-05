"""Public names for score normalization and the adaptive gate."""

from .hybrid import (
    adaptive_gate_scores,
    adaptive_trustrec_details,
    adaptive_trustrec_scores,
    fixed_hybrid_details,
    fixed_hybrid_scores,
    normalize_scores,
    percentile_normalize,
)

normalize_component_scores = percentile_normalize
fixed_hybrid = fixed_hybrid_scores
adaptive_gate = adaptive_trustrec_scores
compute_adaptive_gate = adaptive_trustrec_scores

__all__ = [
    "adaptive_gate",
    "adaptive_gate_scores",
    "adaptive_trustrec_details",
    "adaptive_trustrec_scores",
    "compute_adaptive_gate",
    "fixed_hybrid",
    "fixed_hybrid_details",
    "fixed_hybrid_scores",
    "normalize_component_scores",
    "normalize_scores",
    "percentile_normalize",
]
