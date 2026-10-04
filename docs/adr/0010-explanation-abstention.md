# ADR-0010: Explanation Abstention and Faithfulness

Status: Accepted
Date: 2026-10-03

## Decision

An explanation can be `supported`, `insufficient_support`, or `unavailable`. The system must show positive and negative evidence when they conflict. It can refuse a strong claim. Test faithfulness by removing cited evidence and recomputing the local aspect score. Keep the candidate set and original normalization fixed.

## Consequences

Do not report explanation coverage without explanation correctness. MF or graph scores can keep an item high. The explanation must state when aspect evidence is only one score part.
