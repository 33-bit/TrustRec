"""Recommendation models and shared ranking contracts."""

from .bpr_mf import BPRMFRanker, rank_bpr_mf
from .contracts import (
    CandidateSet,
    RankedItem,
    RankingResult,
    build_candidate_set,
    coerce_interactions,
    first_interactions_before_cutoff,
)
from .fixed_hybrid import FixedHybridRanker, rank_fixed_hybrid
from .hybrid import (
    HybridScore,
    adaptive_trustrec_details,
    adaptive_trustrec_scores,
    fixed_hybrid_details,
    fixed_hybrid_scores,
    percentile_normalize,
)
from .item_knn import ItemKNNRanker, cosine_similarity, rank_item_knn
from .popularity import PopularityRanker, rank_popularity
from .trustrec import TrustRecRanker, rank_trustrec

__all__ = [
    "BPRMFRanker",
    "CandidateSet",
    "FixedHybridRanker",
    "HybridScore",
    "ItemKNNRanker",
    "PopularityRanker",
    "RankedItem",
    "RankingResult",
    "TrustRecRanker",
    "adaptive_trustrec_details",
    "adaptive_trustrec_scores",
    "build_candidate_set",
    "coerce_interactions",
    "cosine_similarity",
    "first_interactions_before_cutoff",
    "fixed_hybrid_details",
    "fixed_hybrid_scores",
    "percentile_normalize",
    "rank_bpr_mf",
    "rank_fixed_hybrid",
    "rank_item_knn",
    "rank_popularity",
    "rank_trustrec",
]
