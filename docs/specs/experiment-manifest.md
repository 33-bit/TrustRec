# Experiment Manifest Contract

Each run writes one JSON manifest beside its metrics:

```json
{
  "run_id": "t0-bpr-seed07",
  "snapshot_id": "video_games-pilot-...",
  "protocol_version": "evaluation-v1",
  "model": "bpr_mf",
  "config_path": "configs/experiments/b2_bpr_mf.toml",
  "config_sha256": "...",
  "dataset_hash": "...",
  "cutoffs": {"t0": "...", "t1": "..."},
  "seed": 7,
  "code_revision": "...",
  "metrics_path": "...",
  "status": "complete"
}
```

`status` is `planned`, `running`, `complete`, or `failed`. A failed run retains its error and environment metadata. Never overwrite a completed manifest; create a new `run_id`.

