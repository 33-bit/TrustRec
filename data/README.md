# Data Directory

Raw and generated data are external artifacts. Do not commit review files, model weights, caches, or large Parquet outputs.

- `raw/` — downloaded source files or range-scan caches; ignored.
- `interim/` — normalized temporary tables; ignored.
- `processed/` — snapshot Parquet and model-ready tables; ignored.
- `manifests/` — snapshot provenance JSON; ignored by default, but copy small decision manifests into `docs/tasks/` when they are part of a reviewable result.

Use `scripts/build_snapshot.py` and record the exact command, source URL, byte size, source hash/window hashes, cutoff, and snapshot ID.

