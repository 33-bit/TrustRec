# ADR-0009: Score Normalization and Adaptive Gate

- **Status:** Accepted as testable design
- **Date:** 2026-10-03

## Decision

Normalize component scores as within-candidate percentile ranks before combining them. Compare a fixed hybrid against the adaptive gate. The gate increases the aspect contribution when user history is short and item evidence support is high; it falls back toward MF/graph when support is weak.

## Consequences

The gate is a hypothesis, not a guaranteed improvement. Its parameters are selected on validation only. Score components and gate values are persisted for faithfulness analysis.

