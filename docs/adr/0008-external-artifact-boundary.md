# ADR-0008: External Data and Artifact Boundary

- **Status:** Accepted
- **Date:** 2026-10-03

## Decision

Raw Amazon files, model weights, caches, Parquet snapshots, generated figures, and large reports stay outside normal Git history. The repository keeps small manifests, schemas, checksums, sample worksheets, and documentation. Ignored artifacts remain reproducible through source URLs, hashes, config, and scripts.

## Consequences

Any result copied into a report must link to a manifest and command. A local file path without a source hash is not reproducible evidence.

