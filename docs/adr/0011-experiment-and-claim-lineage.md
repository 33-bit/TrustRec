# ADR-0011: Experiment and Claim Lineage

Status: Accepted
Date: 2026-10-03

## Decision

Give each experiment a unique run ID and manifest. The manifest contains the snapshot, protocol, configuration, seed, code revision, model or prompt version, dataset hash, metrics path, and status. Reports reference run IDs.

## Consequences

Create a new run when a prompt, threshold, or dependency changes. Keep completed manifests unchanged. Keep failed runs for audit.
