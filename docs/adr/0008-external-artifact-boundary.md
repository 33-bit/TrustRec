# ADR-0008: External Data and Artifact Boundary

Status: Accepted
Date: 2026-10-03

## Decision

Keep raw Amazon files, model weights, caches, Parquet snapshots, generated figures, and large reports outside normal Git history. Keep small manifests, schemas, checksums, sample sheets, and documentation in the repository. Rebuild ignored artifacts from source URLs, hashes, configurations, and scripts.

## Consequences

Each report result must link to a manifest and command. A local path without a source hash is not reproducible evidence.
