# ADR-0004: Repository and Reproducibility Boundaries

Status: Accepted
Date: 2026-10-02

## Decision

Version source code, schemas, small fixtures, manifests, configurations, and documentation. Keep raw reviews, models, caches, and reports as external artifacts with hashes. Use `make check` as the shared quality gate. Each experiment records its seed, cutoff, dataset hash, model hash, and configuration.

## Consequences

Contributors can run tests without production data. To reproduce an experiment, obtain the external artifact named in its manifest. Do not rely on an untracked local path.
