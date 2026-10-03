# T1.2 — Snapshot and Parquet Subset

**Status:** Complete for the Video Games pilot  
**Snapshot:** `video_games-pilot-9e665a862c1a`  
**Manifest:** `data/manifests/video_games-pilot-9e665a862c1a.json`

This is a **development pilot, not the final benchmark snapshot**. Entity filtering for software games versus accessories and the final benchmark eligibility decision remain open.

## Build rule

The builder reads three fixed 32 MiB ranges from the Video Games review JSONL, removes duplicate stable review IDs, sorts by `(timestamp, user_id, parent_asin)`, and keeps at most 200,000 rows. It then scans the metadata JSONL sequentially in 32 MiB ranges and keeps only records whose `parent_asin` occurs in the review subset. No raw source file is stored in the repository.

## Output

| Artifact | Rows | Location |
| --- | ---: | --- |
| interactions | 168,470 | `data/processed/video_games/video_games-pilot-9e665a862c1a/interactions.parquet` |
| review_texts | 168,470 | `data/processed/video_games/video_games-pilot-9e665a862c1a/review_texts.parquet` |
| item_metadata | 37,789 | `data/processed/video_games/video_games-pilot-9e665a862c1a/item_metadata.parquet` |
| train_interactions | 134,776 | `data/processed/video_games/video_games-pilot-9e665a862c1a/train_interactions.parquet` |
| fit_interactions | 151,623 | `data/processed/video_games/video_games-pilot-9e665a862c1a/fit_interactions.parquet` |
| validation_targets | 6,659 | `data/processed/video_games/video_games-pilot-9e665a862c1a/validation_targets.parquet` |
| test_targets | 6,419 | `data/processed/video_games/video_games-pilot-9e665a862c1a/test_targets.parquet` |

All 37,789 item IDs in the subset matched a metadata record. The snapshot uses global timestamp quantiles near 80% and 90% as `T0` and `T1`; the exact UTC cutoffs are stored in the manifest. The canonical dataset hash is `9e665a862c1a760c886faa016cc8e2f90d06105f39d40e0328bde55baed584ca`.

## Reproducibility check

The builder was run twice with identical category, byte-window, maximum-row, and metadata-mode arguments. The second run produced the same dataset hash and the same SHA-256 values for all seven Parquet artifacts. Only the manifest's `created_at` field changes between runs.

## Boundary for T1.3

The subset now has stable IDs, source timestamps, and explicit cutoffs. T1.3 should add assertions that no feature/evidence timestamp reaches its cutoff, no target review ID enters a feature table, and the same candidate/user sets are used across models.
