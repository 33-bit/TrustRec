# Project status and evidence boundary

This record describes the project on 2026-10-04. A pilot is a limited dataset for development. A design target is not a measured result.

## Completed work

The repository contains tools, contracts, ADRs, specs, plans, and tasks. The team created a full Video Games profile and a deterministic filtered benchmark snapshot. Snapshot code now validates IDs, cutoffs, ordering, metadata scope, repeated user-item pairs, duplicate groups, and source hashes. Leakage checks cover timestamps, target IDs, metadata, evaluation sets, and frozen pseudo-test use.

The team completed T1.1, T1.2, and T1.3 for the full manifest `video_games-full-d6c4efeb74aa`. The scan reads 4,624,615 reviews and 137,269 metadata rows. The final tables contain 1,717,097 interactions, 49,749 items, 1,031,539 users, and 100% metadata coverage. The leakage checker passed.

The team completed T2.1 by rebuilding and validating the 100-unit LLM development pilot from pre-`T0` reviews in the full snapshot. It uses `development_pilot` and `llm_silver`. The CSV and JSONL artifacts share exact provenance and evidence checks.

The team completed T2.2 with an independent frozen `llm_pseudo_test` split from the full snapshot. It contains 100 units and 189 aspect labels. The manifest excludes pilot reviews and duplicate groups. It locks source, prompt, model, and artifact hashes. The validator reported 100 valid units.

## Open work

NLP consistency results are recorded in [T2.3](../tasks/t2_3-nlp-baselines.md).
T4.1 now provides snapshot-scoped evidence profiles, shared hybrid
normalization, H0, and T0. Explanation audit results, ranking metrics,
faithfulness, bootstrap intervals, latency, and demo results are pending.

No final human gold dataset exists. Manual labeling is not required for the Amazon Reviews 2023 corpus. Recommendation results must use real Amazon temporal interactions and ratings.
