# ADR-0004: Repository and Reproducibility Boundaries

- **Status:** Accepted
- **Date:** 2026-10-02

## Decision

Source code, schemas, small fixtures, manifests, configs, and documentation are versioned. Raw reviews, generated models, caches, and reports are external artifacts referenced by hashes. `make check` is the shared quality gate, and every experiment records seed, cutoff, dataset hash, model hash, and configuration.

## Consequences

Contributors can run tests without downloading production data. Reproducing an experiment requires obtaining the external artifact identified in its manifest rather than relying on an untracked local path.

