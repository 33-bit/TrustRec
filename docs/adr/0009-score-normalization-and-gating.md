# ADR-0009: Score Normalization and Adaptive Gate

Status: Accepted as testable design
Date: 2026-10-03

## Decision

Normalize each score as a percentile rank within the user candidate set. Compare a fixed hybrid with the adaptive gate. The gate gives more weight to aspects when history is short and evidence support is high. It gives more weight to MF and graph scores when support is weak.

## Consequences

The gate is a hypothesis. It is not a guaranteed improvement. Select its parameters on validation only. Store score parts and gate values for faithfulness analysis.
