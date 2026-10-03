# TrustRec Foundation Design

**Status:** Approved for repository setup  
**Date:** 2026-10-02  
**Source:** `TrustRec-Design (1).md`

## Goal and scope

This foundation makes the TrustRec capstone reproducible before model work starts. It defines repository boundaries, snapshot-aware contracts, quality gates, and the order in which data, NLP, ranking, explanation, evaluation, and demo work will be added. It does not claim a chosen Amazon category, a completed dataset profile, or any achieved metric.

The first benchmark targets top-K recommendation for known items. A valid prediction uses only events before a global cutoff. Rating 4+ is the positive ranking signal; unseen items are not treated as confirmed negatives. Explanations use aspect evidence that can be traced to source review IDs and timestamps.

## Architecture

The pipeline is split into six contracts: snapshot construction, aspect evidence extraction, profile aggregation, recommender fitting, explanation serving, and temporal evaluation. Each artifact carries a `snapshot_id`; feature timestamps must be earlier than that snapshot cutoff. Offline jobs create Parquet or sparse artifacts, while the Streamlit app reads prepared artifacts and never runs the full corpus pipeline inside a request.

Production modules live under `src/trustrec/`. The `schemas` package contains standard-library dataclasses so contract tests can run without downloading data-science dependencies. `configs/protocol.toml` is the versioned default protocol; experiment-specific overrides must be recorded with the output manifest.

## Manual annotation data policy

Manual labels are held-out evaluation data and never enter recommendation, aspect-extraction, or sentiment-model training. Amazon Reviews 2023 by McAuley Lab remains the main training/recommendation source. A development/pilot manual subset may refine guidelines, prompts, thresholds, or model settings; a separate final manual test subset is frozen before tuning and is used only to evaluate aspect extraction, sentiment, and explanation quality. Manual labeling is sampled and does not cover the entire corpus.

## Required contracts

- `SnapshotManifest` records cutoff, source URI, dataset hash, creation time, and row counts.
- `EvidenceRef` records review ID, item, aspect, source text, sentiment, confidence, and source timestamp.
- `RecommendationRecord` records rank, total score, component scores, evidence, and explanation status.
- Evaluation manifests record candidate rules, target rules, seed, model hash, and dataset hash.

## Quality gates

Every new pipeline module needs deterministic unit tests and at least one contract test. Leakage checks must assert that feature and evidence timestamps precede the cutoff, target review IDs are absent from features, and models compare the same user/candidate set. The repository gate is `make check`; the structural gate is `make validate`.

## Non-goals

The core project does not detect fraudulent reviews, expose real reviewer identity, treat PageRank as a trust label, or present ranking scores as purchase probabilities. Node2vec, NCF, diversification, and stronger NLP models remain optional extensions after the baseline ladder and evaluation protocol are stable.
