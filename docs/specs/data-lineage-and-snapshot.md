# Data Lineage and Snapshot Contract

Lineage records where data comes from. A snapshot is a versioned view for time-based analysis. The pipeline follows this path:

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

Each artifact carries `snapshot_id`, a schema version, source and window hashes, a creation time, and a configuration hash. A local path does not provide full lineage.

The required tables are `interactions`, `review_texts`, `aspect_evidence`, `user_profiles`, `item_profiles`, `recommendations`, `train_interactions`, `fit_interactions`, `validation_targets`, and `test_targets`. Raw tables can contain future events. Feature tables must state their cutoff and pass leakage tests. Leakage means that future or held-out data influences a prediction.

Join reviews with metadata before applying the software-game entity filter. Keep `Video Game`, `Software Download`, `Game`, `Computer Game`, `CD-ROM`, `DVD-ROM`, `Game Cartridge`, and `Software` item types. Exclude accessories, consoles, controllers, cables, headsets, shipping, packaging, sellers, books, and unrelated hardware. Record the matched and excluded counts.

Record metadata for display as a retrieval snapshot. Do not use it as a historical feature without a timestamp. If ranking or profiling uses metadata, record its timestamp in the run manifest.

Use these ID rules:

- `review_id`: Use SHA-256 of the canonical source record plus category. Truncate only for display.
- `item_id`: Use `parent_asin`. Keep `asin` separately when present.
- `duplicate_group`: Hash normalized lowercase review text with the item ID.
- `snapshot_id`: Use category, profile mode, and the canonical dataset hash prefix.

Sort source rows by `(timestamp, user_id, item_id, review_id)`. For a repeated user-item pair, keep the first event as the benchmark interaction. Later events can appear in later snapshots, but they cannot create another target for the same pair.
