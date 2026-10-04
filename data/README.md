# Data Directory

Raw and generated data are external artifacts. Do not commit review files, model weights, caches, or large Parquet files.

- `raw/`: Downloaded source files or range-scan caches. Git ignores this directory.
- `interim/`: Normalized temporary tables. Git ignores this directory.
- `processed/`: Snapshot Parquet and model-ready tables. Git ignores this directory.
- `manifests/`: Snapshot provenance JSON. Git ignores this directory by default. Copy small decision manifests into `docs/tasks/` when they are part of a reviewable result.

Use `scripts/build_snapshot.py`. Record the exact command, source URL, byte size, source hash or window hashes, cutoff, and snapshot ID.
