# TrustRec Task Backlog

| ID | Status | Task | Depends on | Suggested owner | Done when |
| --- | --- | --- | --- | --- | --- |
| T0.1 | done | Install harness and run structural checks |: | all | `make validate` and contract tests pass |
| T0.2 | done | [Lock Python/dependency versions](t0_2-environment.md) | T0.1 | owner 1 | [`requirements.lock`](../../requirements.lock), `.python-version`, and [`t0_2-install.log`](t0_2-install.log) are recorded |
| T0.3 | done | Lock design/spec/ADR/folder map | T0.1 | all | docs index, contracts, plans, and folder READMEs exist |
| T1.1 | done | Profile full Video Games category | T0.1 | owner 1 | full source profile records counts, missingness, scope, and retention |
| T1.2 | done | Build full snapshot manifest and Parquet subset | T1.1 | owner 1 | full manifest, source hashes, artifact hashes, and temporal tables are recorded |
| T1.3 | done | Add leakage assertions to the full snapshot | T1.2 | owner 2 | full train, fit, validation, and recommendation test tables pass leakage checks |
| T2.1 | done: full-source development pilot | Run the 100-unit LLM development pilot before `T0` | T1.1 | owner 2 | all units use the full snapshot and have exact cutoff and LLM provenance |
| T2.2 | done | [Freeze an independent LLM pseudo-test split and lock the model, prompt, and version manifest](t2_2-llm-pseudo-test.md) | T2.1 | owners 2–3 | [`t2_2_llm_pseudo_test.manifest.json`](t2_2_llm_pseudo_test.manifest.json) freezes 100 units and blocks tuning |
| T2.3 | done | [Implement aspect and sentiment baselines](t2_3-nlp-baselines.md) | T2.2 | owner 2 | [`t2_3_nlp_baselines.manifest.json`](t2_3_nlp_baselines.manifest.json), predictions, consistency metrics, and error samples are exported |
| T3.1 | done | [Implement popularity and item kNN](t3_1-ranking-baselines.md) | T1.2 | owner 3 | shared candidate contract; output: `src/trustrec/recommenders/{contracts,popularity,item_knn}.py` |
| T3.2 | done | Implement BPR MF and PPR | T3.1 | owner 3 | seed-controlled artifacts and component scores exist |
| T4.1 | done | [Implement evidence aggregation and adaptive gate](t4_1-evidence-aggregation.md) | T2.3,T3.2 | owner 2 | fixed and adaptive hybrids share normalization and tuning budget |
| T4.2 | done | [Implement explanation and faithfulness checks](t4_2-explanation-faithfulness.md) | T4.1 | owner 2 | claims cite evidence and removal tests are logged |
| T5.1 | done | [Run baseline, ablation, and sparse-history evaluation](t5_1-evaluation.md) | T4.2 | owner 1 | metrics, bootstrap intervals, and slice counts are versioned |
| T5.2 | done | [Build the five-minute Streamlit demo](t5_2-streamlit-demo.md) | T4.2 | owner 3 | demo follows `docs/specs/demo-contract.md` and the AppTest smoke run passes |
| T5.3 | done | [Package report and reproducibility bundle](t5_3-report-bundle.md) | T5.1,T5.2 | all | every claim links to a manifest and command |
| T6.1 | done: code and synthetic full-size run | [Make B0–B3 and the evaluator run on the full snapshot](t6_1-scalable-rankings.md) | T3.2,T5.1 | owner 3 | `scripts/run_rankings_batch.py` rows pass `scripts/run_evaluation.py`; ADR-0014 records the changes |
| T6.2 | done: dictionary evidence and gpt-6-luna LLM labels for all t1 reviews, curated | [Label aspect evidence for the full review corpus before each cutoff](t6_2-corpus-evidence.md) | T2.3 | owner 2 | evidence rows for every in-scope review feed `scripts/build_evidence_profiles.py` |
| T6.3 | done: H0 and T0 tuned on `t0` and run on `t1` with gpt-6-luna evidence | Batch H0 and T0 rows from full-catalog component scores | T6.1,T6.2 | owner 3 | H0 and T0 JSON Lines rows pass `scripts/run_evaluation.py` |
| T6.4 | in progress: B0–B5 (no B4) and H0/T0 under ADR-0009 and ADR-0017 evaluated; ADR-0017 H0 is best, +0.0007 over B5, aspect effect small and not robust to the ablation rule; LLM and dictionary evidence tie (see task) | [Run the full Video Games evaluation for every model and seed](t6_4-baseline-results.md) | T6.1,T6.3 | owner 1 | metrics, manifests, and the T5.3 bundle use real ranking files |
| T7.1 | in progress: first run done; 99.9% of supported claims are "mixed" under the current direction rule (see task) | [Generate explanations for the ADR-0017 H0 top-K on the real `t1` snapshot](t7_1-explanations.md) | T6.4 | owner 2 | coverage and abstention (RQ5) are reported by history, popularity, and item age, with a manifest |
| T7.2 | todo | Run the evidence-removal faithfulness test on real explanations | T7.1 | owner 2 | cited versus random removal (RQ4) is reported with intervals |
| T7.3 | todo | Audit a sample of real explanations with the explanation audit contract | T7.1 | owner 2 | audit fields and LLM-judge provenance are recorded (RQ4) |
| T7.5 | todo | Update the report bundle and demo to the real-data results | T6.4,T7.1 | owners 1, 3 | bundle claims follow ADR-0018 and the demo uses the real snapshot |
