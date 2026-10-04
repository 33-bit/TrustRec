"""Recommendation models and shared ranking contracts."""

from .contracts import (
    CandidateSet,
    RankedItem,
    RankingResult,
    build_candidate_set,
    coerce_interactions,
    first_interactions_before_cutoff,
)
from .item_knn import ItemKNN, ItemKNNRanker, cosine_similarity, rank_item_knn
from .popularity import MostPopularRanker, PopularityRanker, rank_most_popular, rank_popularity

__all__ = [
    "CandidateSet",
    "ItemKNNRanker",
    "ItemKNN",
    "MostPopularRanker",
    "PopularityRanker",
    "RankedItem",
    "RankingResult",
    "build_candidate_set",
    "coerce_interactions",
    "cosine_similarity",
    "first_interactions_before_cutoff",
    "rank_item_knn",
    "rank_most_popular",
    "rank_popularity",
]
