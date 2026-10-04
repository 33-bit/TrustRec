# Phase 5 Plan: Evaluation, Demo, and Report

Create a result package with traceable records and a five-minute demo. A run manifest records the inputs and outputs of an experiment. Each reported claim links to that record.

## Tasks

1. Run baselines, ablations, sparse-history groups, popularity groups, and controlled noise tests.
2. Read `docs/specs/evaluation-contract.md` for the metrics and comparison rules.
3. Run three final seeds and paired bootstrap intervals on the locked user set.
4. Export metric tables, figures, latency, resource use, error cases, and manifests.
5. Build the Streamlit selection, ranking, evidence, and priority-change flow.
6. Prepare a demo fallback.
7. Map each report claim to a run ID.

## Exit rules

The report separates proposals, measurements, and limitations. Read test results only after configuration and thresholds are locked. Each result links to a manifest.
