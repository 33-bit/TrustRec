# Evaluation and ablation contract

Use global time boundaries `T0 < T1`. Training uses events before `T0`. Validation targets use `[T0, T1)`. Final fitting uses events before `T1`. Test targets use events at or after `T1`.

Candidates are known items that the user did not review before the cutoff. A target must belong to the catalog before its cutoff. Use the same users, candidates, and targets for every compared model.

NDCG@10 is the primary metric. Also report Recall@K, Precision@K, coverage, diversity, popularity slices, sparse-history slices, explanation coverage, latency, memory, and fit time. Report eligible users and exclusion reasons.

Use a paired user bootstrap for uncertainty. Use three fixed seeds for random models. Run ablations with the same data, candidate set, and tuning budget.

Stop evaluation when a feature or evidence timestamp reaches its cutoff. Stop when a target review ID appears in features or evidence. Stop when post-cutoff metadata, different evaluation sets, or frozen pseudo-test tuning is detected.

Amazon Reviews 2023 by McAuley Lab supplies recommendation interactions and ratings. The temporal recommendation test is `recommendation_test`. LLM labels use `development_pilot` or frozen `llm_pseudo_test`. They are not human gold labels.
