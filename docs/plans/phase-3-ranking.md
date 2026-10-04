# Phase 3 Plan: Ranking Core

Implement ranking baselines on the same snapshot contract. A baseline is a model used for comparison. Each model must rank the same candidate set.

## Tasks

1. Implement most-popular and item-kNN baselines.
2. Implement BPR MF with pre-cutoff positive interactions and unknown negatives.
3. Build the graph from positive user-item edges.
4. Implement personalized PageRank with a fallback for nodes that have no edges.
5. Implement aspect-only scores and percentile normalization.
6. Store score parts, candidate IDs, seeds, and model hashes.

## Exit rules

All B0–B4 models rank identical user and candidate sets. They produce comparable manifests. No model reads future target text or metadata.
