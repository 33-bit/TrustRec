# TrustRec Delivery Roadmap

## Phase 0 — Foundation (current)

Repository harness, contracts, ADRs, protocol config, structural tests, and contributor guidance. Exit when `make validate` and the contract tests pass.

## Phase 1 — Data and snapshots

Profile Video Games and one fallback category, freeze the category decision, normalize schemas, create Parquet subsets, and publish train/validation/test manifests. Exit when timestamp and retention reports are reproducible.

## Phase 2 — Aspect NLP

Pilot the eight-aspect ontology, label the gold set, implement dictionary and TF-IDF/SVM baselines, calibrate confidence, and store offsets. Exit when the locked NLP protocol has metrics and error examples.

## Phase 3 — Ranking core

Implement popularity, item kNN, BPR MF, PPR, percentile normalization, and fallback branches. Exit when all core baselines run on the same snapshot and candidate set.

## Phase 4 — TrustRec and explanations

Aggregate weighted evidence, implement the adaptive gate, render traceable explanations, and run faithfulness checks. Exit when every recommendation has an auditable component breakdown and explanation status.

## Phase 5 — Evaluation and demo

Run ablations, sparse-history slices, noise stress tests, bootstrap intervals, and the Streamlit five-minute flow. Exit when artifacts, figures, and report claims point to immutable manifests.

