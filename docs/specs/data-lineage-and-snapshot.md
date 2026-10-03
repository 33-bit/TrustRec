# Data Lineage and Snapshot Contract

## Lineage chain

```text
source URL + retrieval date
  → byte/range or full scan provenance
  → normalized JSONL records
  → stable IDs + schema validation
  → snapshot cutoffs T0/T1
  → train/fit/validation/test tables
  → model/profile/evidence artifacts
  → recommendations + explanations + result manifest
```

Every artifact must carry `snapshot_id`, schema version, source hash or source window hashes, creation time, and configuration hash. A local path is not sufficient provenance.

## Required snapshot tables

`interactions`, `review_texts`, `aspect_evidence`, `user_profiles`, `item_profiles`, `recommendations`, `train_interactions`, `fit_interactions`, `validation_targets`, and `test_targets`. Raw tables may contain future events; feature tables must state their cutoff and pass leakage checks.

Metadata used only to determine entity scope or display fields is recorded as a retrieval snapshot. It must not silently become a historical feature. Any metadata field used for ranking or profiling needs a timestamp-valid lineage decision in the run manifest.

## Stable ID rules

- `review_id`: SHA-256 of canonical source record plus category, truncated only for display if needed.
- `item_id`: source `parent_asin`; retain `asin` separately when present.
- `duplicate_group`: hash of normalized lower-cased review text within item scope.
- `snapshot_id`: category, profile mode, and canonical dataset hash prefix.
