# T1.3: full leakage checks

Status: Complete for the full benchmark snapshot.

Manifest: [`t1_2_full_snapshot_manifest.json`](t1_2_full_snapshot_manifest.json).

The leakage command reads the full train, fit, validation target, recommendation test, and item metadata tables. It confirms that `T0` is earlier than `T1`, required IDs and timestamps exist, training rows precede `T0`, fit rows precede `T1`, and train and fit review IDs do not appear in target tables.

The command also checks feature and evidence tables when a manifest includes them. It checks shared evaluation sets and frozen pseudo-test restrictions when those records exist.

Run the leakage command from the repository root:

```bash
PYTHONPATH=src python3 scripts/check_snapshot_leakage.py docs/tasks/t1_2_full_snapshot_manifest.json
```

The leakage command passed for `video_games-full-d6c4efeb74aa` on 2026-10-04. The full snapshot is eligible for recommendation model evaluation. NLP and explanation artifacts still need their own cutoff checks.
