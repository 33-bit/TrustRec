# Project status and evidence boundary

This record describes the project on 2026-10-04. A pilot is a limited dataset for development. A design target is not a measured result.

## Completed work

The repository contains tools, contracts, ADRs, specs, plans, and tasks. The team created a full Video Games profile and a deterministic filtered benchmark snapshot. Snapshot code now validates IDs, cutoffs, ordering, metadata scope, repeated user-item pairs, duplicate groups, and source hashes. Leakage checks cover timestamps, target IDs, metadata, evaluation sets, and frozen pseudo-test use.

The team completed T1.1, T1.2, and T1.3 for the full manifest `video_games-full-d6c4efeb74aa`. The scan reads 4,624,615 reviews and 137,269 metadata rows. The final tables contain 1,717,097 interactions, 49,749 items, 1,031,539 users, and 100% metadata coverage. The leakage checker passed.

The team completed T2.1 by rebuilding and validating the 100-unit LLM development pilot from pre-`T0` reviews in the full snapshot. It uses `development_pilot` and `llm_silver`. The CSV and JSONL artifacts share exact provenance and evidence checks.

The team completed T2.2 with an independent frozen `llm_pseudo_test` split from the full snapshot. It contains 100 units and 189 aspect labels. The manifest excludes pilot reviews and duplicate groups. It locks source, prompt, model, and artifact hashes. The validator reported 100 valid units.

## Completed evaluation work

The team completed T5.1 with pure ranking metrics, paired user bootstrap
intervals, temporal case construction, popularity and sparse-history slices,
lineage validation, and a JSON Lines evaluation runner. Fixture tests cover
the metric formulas, set equality, exclusion reasons, and manifest hashes.

The full-category evaluation command is recorded in [T5.1](../tasks/t5_1-evaluation.md).
Generated Parquet tables and model ranking files remain external to Git, so
full Video Games metric values are not stored in this repository.

## Completed demo work

The team completed T5.2 with a prepared-artifact service and a Streamlit view.
The view selects a low-history user, snapshot, model, K, and temporary aspect
priorities. It compares baseline and TrustRec rankings, shows score parts,
shows source evidence and support states, and keeps lineage fields visible.
The checked-in fallback bundle has SHA-256
`841ba49dd421b1f350bf370b15dfceff5550c1edd5bbc7d73f80c5daefea3285`.
The fallback is synthetic and does not support a production metric claim.

The team completed T5.3 with a deterministic report packager. The package
records claim pointers, run manifests, commands, source hashes, and external
artifact states. The default report includes the snapshot and NLP evidence,
the synthetic demo marker, and the missing recommendation result as a limit.

## Open work

NLP consistency results are recorded in [T2.3](../tasks/t2_3-nlp-baselines.md).
T4.1 provides snapshot-scoped evidence profiles, shared hybrid normalization,
H0, and T0. T4.2 provides claim selection, abstention, and deterministic
evidence-removal audit artifacts. T5.3 remains open.

No final human gold dataset exists. Manual labeling is not required for the Amazon Reviews 2023 corpus. Recommendation results must use real Amazon temporal interactions and ratings. The full recommendation metric table remains external until the T5.1 ranking inputs are available.
