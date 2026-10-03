# Project Status and Evidence Boundary

**As of:** 2026-10-03

## Completed evidence

- Repository harness, contracts, ADR/spec/plan/task structure.
- Video Games pilot profile and deterministic byte-window snapshot.
- Snapshot leakage checks and hash repeatability.
- AI preliminary annotation pilot and Laya comparison.

## Pilot-only artifacts

`video_games-pilot-9e665a862c1a` is a bounded pilot built from fixed byte windows plus a full metadata scan for the sampled IDs. It is useful for pipeline development and contract checks. It is not the final full-category benchmark and has not passed software-game entity filtering or the final NLP gold/test split.

## Not yet measured

BPR MF, PPR, item-kNN, evidence aggregation, adaptive gate, final NLP metrics, recommendation metrics, faithfulness, bootstrap intervals, latency, and production-like demo results remain implementation/evaluation work. Design target numbers are not results.

