# Recommendation Protocol

The benchmark uses global time boundaries `T0 < T1`. Training uses events before `T0`; validation targets fall in `[T0, T1)`; final fitting uses events before `T1`; test targets fall at or after `T1`. Candidates are known items before the relevant cutoff minus the user’s prior history. A target is a newly observed item with rating 4 or higher.

Primary metrics are NDCG@10 and Recall@10, with K in `{5, 10, 20}`. Report the number of eligible and ineligible users, candidate retention, and target items outside the known catalog. Compare most popular, item kNN, BPR MF, PPR, aspect-only, fixed hybrid, and adaptive TrustRec under equal tuning budgets.

All models use the same users, candidates, targets, seeds, and snapshot. Store dataset hash, model hash, configuration, cutoff timestamps, and bootstrap settings with each result.

