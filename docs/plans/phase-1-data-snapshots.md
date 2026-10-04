# Phase 1 Plan: Data and Snapshots

Create a reproducible snapshot from the category decision. A snapshot is a versioned view for time-based analysis. This phase also separates software games from accessories.

## Tasks

1. Make sure that the source URL, retrieval date, raw byte size, and source hashes are recorded.
2. Join review `parent_asin` to metadata.
3. Classify software games and accessories.
4. Normalize `interactions`, `review_texts`, and metadata fields without changing source text.
5. Derive global `T0` and `T1`.
6. Create train, fit, validation target, and test target tables.
7. Run leakage assertions.
8. Publish a manifest with row counts, retention, and metadata coverage.

## Exit rules

The same command produces the same canonical and Parquet hashes. The report lists scope exclusions and user and item counts. It also records target retention, missing fields, and all cutoff timestamps.
