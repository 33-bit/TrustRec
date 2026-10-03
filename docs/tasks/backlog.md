# TrustRec Task Backlog

| ID | Status | Task | Depends on | Suggested owner | Done when |
| --- | --- | --- | --- | --- | --- |
| T0.1 | done | Install harness and run structural checks | — | all | `make validate` and contract tests pass |
| T0.2 | planned | Lock Python/dependency versions | T0.1 | owner 1 | environment file and install log are recorded |
| T0.3 | done | Lock design/spec/ADR/folder map | T0.1 | all | docs index, contracts, plans, and folder READMEs exist |
| T1.1 | done — pilot | Profile candidate Amazon categories | T0.1 | owner 1 | category decision memo includes counts, missingness, and retention |
| T1.2 | done — pilot | Build snapshot manifest and Parquet subset | T1.1 | owner 1 | rerun produces identical hashes from the same source |
| T1.3 | done | Add leakage assertions | T1.2 | owner 2 | future timestamps and target review IDs fail tests |
| T2.1 | in progress — development pilot | Pilot aspect guideline and 100 labels | T1.1 | owner 2 | manual development subset is reviewed without entering model training |
| T2.2 | planned | Label and freeze manual final test split | T2.1 | owners 2–3 | grouped frozen final test manifest, overlap agreement, and adjudication exist |
| T2.3 | planned | Implement aspect/sentiment baseline | T2.2 | owner 2 | F1, coverage, and error samples are exported |
| T3.1 | planned | Implement popularity and item kNN | T1.2 | owner 3 | both rank on the shared candidate contract |
| T3.2 | planned | Implement BPR MF and PPR | T3.1 | owner 3 | seed-controlled artifacts and component scores exist |
| T4.1 | planned | Implement evidence aggregation and adaptive gate | T2.3,T3.2 | owner 2 | fixed and adaptive hybrids share normalization and tuning budget |
| T4.2 | planned | Implement explanation and faithfulness checks | T4.1 | owner 2 | claims cite evidence and removal tests are logged |
| T5.1 | planned | Run baseline, ablation, and sparse-history evaluation | T4.2 | owner 1 | metrics, bootstrap intervals, and slice counts are versioned |
| T5.2 | planned | Build the five-minute Streamlit demo | T4.2 | owner 3 | demo follows `docs/specs/demo-contract.md` |
| T5.3 | planned | Package report and reproducibility bundle | T5.1,T5.2 | all | every claim links to a manifest and command |
