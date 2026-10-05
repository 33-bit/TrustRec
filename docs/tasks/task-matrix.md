# TrustRec Task Matrix

This table maps each workstream to its code, output, and research question. RQ means research question. Read [the terminology guide](../reference/terminology.md) for metric and model terms.

| Workstream | Primary files | Artifact | Evidence for acceptance | Related RQ |
| --- | --- | --- | --- | --- |
| Source profile | `scripts/profile_categories.py` | Category memo and profile JSON | Source bytes, missing fields, retention | Feasibility |
| Snapshot | `scripts/build_snapshot.py` | `t1_2_snapshot_manifest.json` and local Parquet tables | Repeatable hashes | All |
| Leakage | `src/trustrec/evaluation/leakage.py` | `t1_3-leakage-report.md` | Timestamp and ID assertions | All |
| Annotation | `configs/aspects.toml`, `docs/tasks/*annotation*` | LLM pilot and frozen pseudo-test | Provenance, offsets, and split roles | RQ3/RQ4 |
| NLP baseline | `src/trustrec/nlp/` | [`t2_3-nlp-baselines.md`](t2_3-nlp-baselines.md), ignored predictions, metrics, and error samples | Aspect, sentiment, evidence-span consistency, and coverage | RQ1/RQ3 |
| Collaborative ranking | `src/trustrec/recommenders/` | Model artifact and scores | Same candidates | RQ1 |
| Graph ranking | `src/trustrec/graph/` | PPR scores and graph manifest | Popularity comparison | RQ1 |
| Evidence profiles | `src/trustrec/explanations/`, `scripts/build_evidence_profiles.py` | Snapshot-scoped profile artifact | Support, duplicate, cutoff, and conflict tests | RQ2/RQ3 |
| Hybrid gate | `src/trustrec/recommenders/`, `scripts/run_hybrid.py` | H0 and T0 score artifacts | Shared normalization, candidate set, and validation budget | RQ2 |
| Faithfulness | `src/trustrec/explanations/`, `docs/specs/explanation-audit-contract.md` | Removal audit | Contribution change and abstention | RQ4 |
| Evaluation | `src/trustrec/evaluation/` | Metrics, bootstrap intervals, groups | Manifest lineage | All |
| Demo/report | `app/`, `reports/` | Demo and figures/report | Five-minute script | RQ4 |
