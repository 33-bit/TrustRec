# Phase 3 Plan — Ranking Core

**Goal:** Implement comparable ranking baselines on the same snapshot contract.

## Tasks

1. Implement most-popular and item-kNN baselines.
2. Implement BPR MF with snapshot-safe positive interactions and unknown negatives.
3. Implement positive-edge user–item graph and personalized PageRank with dangling-node fallback.
4. Implement aspect-only scores and percentile normalization.
5. Persist component scores, candidate IDs, seeds, and model hashes.

## Exit criteria

All B0–B4 models rank identical user/candidate sets and produce comparable manifests. No model reads future target text or metadata.

