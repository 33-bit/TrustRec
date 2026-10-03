# ADR-0010: Explanation Abstention and Faithfulness

- **Status:** Accepted
- **Date:** 2026-10-03

## Decision

An explanation may be `supported`, `insufficient_support`, or `unavailable`. The system must show conflicting positive/negative evidence and can refuse a strong claim. Faithfulness is tested by removing cited evidence and recomputing the local aspect contribution while holding the candidate set and original normalization fixed.

## Consequences

Explanation coverage is never reported without explanation correctness. A recommendation can remain high because MF or graph score dominates; the explanation must state that aspect evidence is only one component.

