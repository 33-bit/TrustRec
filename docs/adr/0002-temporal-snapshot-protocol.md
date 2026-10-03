# ADR-0002: Global Temporal Snapshots

- **Status:** Accepted
- **Date:** 2026-10-02

## Decision

Train, validation, fit-final, and test data are separated by global timestamp cutoffs. User-level “last review” splits are not the primary benchmark because they can allow future information from other users into feature construction. Every feature, graph edge, vocabulary, profile, and evidence item must have a timestamp earlier than its snapshot cutoff.

## Consequences

Snapshot manifests and leakage assertions are first-class artifacts. Any component fitted before test must be rebuilt from data before the test boundary; post-cutoff text cannot enter an explanation.

