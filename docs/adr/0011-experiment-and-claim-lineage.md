# ADR-0011: Experiment and Claim Lineage

- **Status:** Accepted
- **Date:** 2026-10-03

## Decision

Every experiment receives a unique run ID and manifest containing snapshot, protocol, config, seed, code revision, model/prompt version, dataset hash, metrics path, and status. Reports reference run IDs instead of copying untraceable numbers.

## Consequences

Re-running with a changed prompt, threshold, or dependency creates a new run. Completed manifests are immutable; failed runs remain visible for audit.

