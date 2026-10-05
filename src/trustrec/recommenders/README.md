# Recommenders Module

This module owns popularity, item kNN, BPR MF, score normalization, fixed
hybrid, adaptive gate, and fallbacks.

The T3.1 baselines use `CandidateSet` from `contracts.py`. It contains one
user, one snapshot, one cutoff, the user's history, and the exact candidate
IDs. `PopularityRanker` counts positive ratings before the cutoff.
`ItemKNNRanker` sums cosine similarity to positive history items.
`BPRMFRanker` learns latent user and item vectors from positive events and
sampled unknown items with a fixed random seed.

Both rankers return `RankingResult`. The result contains ranked items,
component scores, configuration, seed, model hash, and a fallback reason.
Both rankers preserve the candidate IDs and sort equal scores by item ID.
They use `no_candidates` when the candidate set is empty. Item kNN uses
`no_positive_history` when it falls back to positive popularity.

BPR MF uses `no_positive_history` for a user without positive history and
falls back to positive popularity. It keeps low-rated history items out of
negative samples. Its result includes the MF score for every candidate.
