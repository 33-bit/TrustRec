# TrustRec Foundation Design

Status: Approved for repository setup. Date: 2026-10-02. Source: `TrustRec-Design (1).md`.

## Goal and scope

This foundation makes the TrustRec capstone reproducible before model work starts. It defines repository boundaries, snapshot-aware contracts, quality gates, and the order in which data, NLP, ranking, explanation, evaluation, and demo work will be added. It does not claim a chosen Amazon category, a completed dataset profile, or any achieved metric.

The first benchmark targets top-K recommendation for known items. A valid prediction uses only events before a global cutoff. Rating 4+ is the positive ranking signal. Unseen items are not confirmed negatives. Explanations use aspect evidence that can be traced to source review IDs and timestamps.

## Architecture

The pipeline is split into six contracts: snapshot construction, aspect evidence extraction, profile aggregation, recommender fitting, explanation serving, and temporal evaluation. Each artifact carries a `snapshot_id`. Feature timestamps must be earlier than that snapshot cutoff. Offline jobs create Parquet or sparse artifacts. The Streamlit app reads prepared artifacts and does not run the full corpus pipeline during a request.

Production modules live under `src/trustrec/`. The `schemas` package contains standard-library dataclasses so contract tests can run without downloading data-science dependencies. `configs/protocol.toml` is the versioned default protocol. Record experiment-specific overrides with the output manifest.

## LLM annotation data policy

Amazon Reviews 2023 by McAuley Lab remains the main source for training and recommendation interactions. An LLM creates aspect, sentiment, and evidence labels. The development pilot uses `development_pilot` and `llm_silver`. It can refine guidelines, prompts, thresholds, and model settings. A separate `llm_pseudo_test` pass is frozen before tuning and supports pseudo-label consistency and explanation checks. The recommendation test uses real Amazon temporal interactions and ratings. LLM labels are not human gold labels, and manual labeling is not required for the full corpus.

## Required contracts

- `SnapshotManifest` records cutoff, source URI, dataset hash, creation time, and row counts.
- `EvidenceRef` records review ID, item, aspect, source text, sentiment, confidence, and source timestamp.
- `RecommendationRecord` records rank, total score, component scores, evidence, and explanation status.
- Evaluation manifests record candidate rules, target rules, seed, model hash, and dataset hash.

## Quality gates

Every new pipeline module needs deterministic unit tests and at least one contract test. Leakage checks must assert that feature and evidence timestamps precede the cutoff, target review IDs are absent from features, and models compare the same user/candidate set. The repository gate is `make check`. The structural gate is `make validate`.

## Non-goals

The core project does not detect fraudulent reviews, expose real reviewer identity, treat PageRank as a trust label, or present ranking scores as purchase probabilities. Node2vec, NCF, diversification, and stronger NLP models remain optional extensions after the baseline ladder and evaluation protocol are stable.
