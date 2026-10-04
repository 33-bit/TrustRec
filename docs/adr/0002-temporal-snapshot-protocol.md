# ADR-0002: Global Temporal Snapshots

Status: Accepted
Date: 2026-10-02

## Decision

Use global timestamp cutoffs for train, validation, final fit, and test data. Do not use a user-level last-review split as the main benchmark. That split can leak future data from other users. Each feature, graph edge, vocabulary, profile, and evidence item must have a timestamp before its snapshot cutoff.

## Consequences

Snapshot manifests and leakage checks are required artifacts. Rebuild each test-dependent component from data before the test boundary. Do not use post-cutoff text in an explanation.
