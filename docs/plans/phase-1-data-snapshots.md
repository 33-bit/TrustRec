# Phase 1 Plan — Data and Snapshots

**Goal:** Turn the category decision into a reproducible, scope-filtered snapshot.

## Tasks

1. Confirm source URL, retrieval date, raw byte size, and source hashes.
2. Join review `parent_asin` to metadata and classify software games versus accessories.
3. Normalize `interactions`, `review_texts`, and metadata fields without changing source text.
4. Derive global `T0` and `T1`; create train, fit, validation target, and test target tables.
5. Run leakage assertions and publish a manifest with row counts, retention, and metadata coverage.

## Exit criteria

The same command produces the same canonical and Parquet hashes. The report includes scope exclusions, user/item counts, target retention, missingness, and all cutoff timestamps.

