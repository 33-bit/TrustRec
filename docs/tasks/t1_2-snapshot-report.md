# T1.2: full snapshot and Parquet tables

Status: Complete for the full Video Games category.

Source: Amazon Reviews 2023 by McAuley Lab, Video Games category.

The full benchmark manifest is `video_games-full-d6c4efeb74aa`. The dataset hash is `0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`. The reviewable copy is [`t1_2_full_snapshot_manifest.json`](t1_2_full_snapshot_manifest.json). The builder streams all source records in 50,000-row batches. It scans metadata before the scope filter. It writes source chunk hashes, source hashes, cutoff values, row counts, and artifact hashes.

The full scan reads 4,624,615 review records and 137,269 metadata records. It keeps 1,737,112 software-game review records before stable-ID deduplication. The final tables contain 1,717,097 interactions, 1,717,097 review texts, 49,749 items, and 1,031,539 users.

The snapshot contains 1,373,677 training interactions, 1,545,387 fit interactions, 95,559 validation targets, and 76,435 recommendation test targets. Metadata coverage is 100%.

The cutoffs are `T0 = 2019-01-13T04:24:39.377Z` and `T1 = 2020-08-31T22:46:09.247Z`. The builder sorts rows by `(timestamp, user_id, item_id, review_id)`. It removes duplicate stable review IDs. It keeps the first event for each repeated user and item pair.

Run the full build from the repository root:

```bash
PYTHONPATH=src python3 scripts/build_full_snapshot.py --category Video_Games --batch-size 50000
```

The command writes generated Parquet files under `data/processed/` and the manifest under `data/manifests/`. Do not commit raw source files or generated Parquet files.

The full snapshot is ready for recommendation model training and temporal evaluation. It does not include aspect labels or model result tables.
