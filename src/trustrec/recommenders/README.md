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

T4.1 adds H0 and T0 in `fixed_hybrid.py` and `trustrec.py`. The shared
normalizer in `hybrid.py` converts every component to percentile ranks inside
the same candidate set. H0 combines MF, graph, and aspect ranks with fixed
weights. T0 uses the history and evidence support gate from the ranking
contract. `HybridScore` keeps normalized values, contributions, gate, and
support for later explanation audits.

## Scale path (ADR-0014)

`matrix.py` builds one `InteractionMatrix` from the first pre-cutoff event per
user-item pair. It holds sparse history and positive matrices, positive item
counts, a data fingerprint, and a catalog hash. Pass an Arrow table or a pandas
frame for the vectorized path. Pass `matrix=` to `fit` to share one matrix
across models.

`base.py` holds `MatrixRanker`. It owns candidate validation, fallbacks, and
the ordering rule. `rank(candidate_set, k)` keeps the `RankingResult`
contract. `recommend(user_id, k)` returns the top K items from the catalog
minus the user's history, with the candidate count and signature used by the
evaluator. `registry.py` builds B0 through B3 from one `BaselineParameters`
object.

BPR MF trains with mini-batch SGD and summed updates. `batch_size` (default
1024) is part of the configuration and model hash. Negative samples are drawn
uniformly from items outside the user's history.
