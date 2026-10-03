# T1.3 — Leakage Assertions

**Status:** Complete for `video_games-pilot-9e665a862c1a` as a development-pilot check; the same assertions must run on the final benchmark snapshot.

The snapshot now separates raw interactions from time-safe feature and target tables:

- `train_interactions` contains events before `T0`.
- `fit_interactions` contains events before `T1`.
- `validation_targets` contains positive, unseen item pairs in `[T0, T1)`.
- `test_targets` contains positive, unseen item pairs at or after `T1`.

`scripts/check_snapshot_leakage.py` verifies that feature timestamps are strictly earlier than their cutoff and that train/fit review IDs are disjoint from validation/test target IDs. The checker ran successfully on the generated Parquet files. Unit tests also cover a future timestamp failure, target-ID overlap failure, and a clean disjoint case.
