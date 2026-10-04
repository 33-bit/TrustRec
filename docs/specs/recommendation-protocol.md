# Recommendation Protocol

The benchmark uses global time boundaries `T0 < T1`. Training uses events before `T0`. Validation targets fall in `[T0, T1)`. Final fitting uses events before `T1`. Test targets fall at or after `T1`.

Candidates are known items before the cutoff, minus the user history. A target is a newly observed item with rating 4 or higher. Read [the evaluation contract](evaluation-contract.md) for metric definitions and rules.

Use the first event for a repeated user-item pair as the benchmark interaction. Later rating changes can appear in later snapshots, but they cannot create a new target for that pair.

Primary metrics are NDCG@10 and Recall@10, with K in `{5, 10, 20}`. Report eligible and ineligible users, candidate retention, and target items outside the catalog. Compare most popular, item kNN, BPR MF, PPR, aspect-only, fixed hybrid, and adaptive TrustRec. Give each model an equal tuning budget.

Use the same users, candidates, targets, seeds, and snapshot for all models. Store dataset and model hashes, configuration, cutoff timestamps, and bootstrap configuration with each result.
