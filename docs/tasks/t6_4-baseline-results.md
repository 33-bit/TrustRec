# Task: T6.4 (partial): Full Video Games baseline results

Status: B0 through B3 are done on the real `t1` test set with seed 7. H0 and
T0 wait for T6.3. BPR MF needs validation tuning (see Findings), and seeds 17
and 27 are not run yet.

## Lineage

- Snapshot: `video_games-full-d6c4efeb74aa`, rebuilt on 2026-11-02 with `scripts/build_full_snapshot.py`.
- Six of seven tables are byte-identical to [`t1_2_full_snapshot_manifest.json`](t1_2_full_snapshot_manifest.json): interactions, review texts, train, fit, validation, and test.
- Row counts, cutoffs, and source chunk hashes also match.
- `item_metadata.parquet` differs at the byte level. The original was built on macOS with Python 3.14; this rebuild ran on Linux with Python 3.13. Because the dataset hash includes this table, the rebuilt dataset hash is `7470fba0e8e05c32995a20e22a76cbc2a94a544328902c79fcf9d7be9208d888`. Ranking and evaluation do not read item metadata.
- Code revision: `7b48685`. Protocol: `evaluation-v2`.
- Candidate set hash: `8a53d55a80b068d11339858818052be56782168dd39d9cd0f8beb3c0e80621f4`.
- Metrics artifact SHA-256: `33f9bb81aae2590de844fdfc14f8268b4a1686f3a34aa0af5f567a3b5435d18d`. The metrics artifact and its manifest are in ignored `reports/generated/`. Ranking files are in ignored `artifacts/rankings/`.

## Commands

```bash
PYTHONPATH=src python3 scripts/run_rankings_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b0_most_popular --cutoff t1 --k 100 --workers 4 --output artifacts/rankings/b0_t1.jsonl
PYTHONPATH=src python3 scripts/run_rankings_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b1_item_knn --cutoff t1 --k 100 --workers 4 --output artifacts/rankings/b1_t1.jsonl
PYTHONPATH=src python3 scripts/run_rankings_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b2_bpr_mf --cutoff t1 --k 100 --workers 4 --seed 7 --output artifacts/rankings/b2_t1_seed07.jsonl
PYTHONPATH=src python3 scripts/run_rankings_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b3_ppr --cutoff t1 --k 100 --workers 4 --output artifacts/rankings/b3_t1.jsonl
PYTHONPATH=src python3 scripts/run_evaluation.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cutoff t1 --model b0_most_popular=artifacts/rankings/b0_t1.jsonl --model b1_item_knn=artifacts/rankings/b1_t1.jsonl --model b2_bpr_mf=artifacts/rankings/b2_t1_seed07.jsonl --model b3_ppr=artifacts/rankings/b3_t1.jsonl --run-id t6-4-video-games-t1-baselines-seed07 --output reports/generated/t6-4-video-games-t1-baselines-seed07.json --manifest reports/generated/t6-4-video-games-t1-baselines-seed07.manifest.json --reference-model b0_most_popular --seed 7
```

Wall times on 4 cores:

| Step | Wall time |
| --- | --- |
| B0 | 28 s |
| B1 | 30 s |
| B2 | 343 s |
| B3 | 593 s |
| Evaluation | 186 s |

## Results

The evaluation has 61,775 eligible users and no exclusions. Intervals are 95%
paired user bootstrap intervals with 2,000 samples and seed 7. The reference
model for differences is B0.

| Model | NDCG@10 | Difference from B0 | Recall@10 | Difference from B0 |
| --- | --- | --- | --- | --- |
| B0 most popular | 0.0268 [0.0259, 0.0277] | | 0.0492 [0.0475, 0.0507] | |
| B1 item kNN | 0.0256 [0.0247, 0.0265] | -0.0012 [-0.0016, -0.0008] | 0.0466 [0.0450, 0.0481] | -0.0026 [-0.0032, -0.0020] |
| B2 BPR MF, seed 7 | 0.0268 [0.0259, 0.0277] | 0.0000 [0.0000, 0.0000] | 0.0492 [0.0475, 0.0507] | 0.0000 [0.0000, 0.0000] |
| B3 PPR | 0.0284 [0.0274, 0.0293] | +0.0015 [+0.0012, +0.0020] | 0.0512 [0.0496, 0.0529] | +0.0021 [+0.0014, +0.0027] |

NDCG@10 by pre-cutoff history slice:

| Slice | Users | B0 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- | --- |
| no_history | 53,413 | 0.0276 | 0.0276 | 0.0276 | 0.0276 |
| 1-2 | 5,464 | 0.0237 | 0.0158 | 0.0237 | 0.0371 |
| 3-5 | 1,546 | 0.0201 | 0.0095 | 0.0201 | 0.0294 |
| >5 | 1,352 | 0.0155 | 0.0044 | 0.0156 | 0.0215 |

## Findings

- Cold start dominates the test set. 54,153 users (88%) have no positive interaction before `T1`, so every personalized baseline falls back to popularity for them. Gains can only come from the remaining 12%.
- B3 PPR is the only baseline that beats popularity. Its gain is largest in the `1-2` slice.
- B1 item kNN is below popularity in every slice that has history.
- B2 BPR MF with the configured settings (`learning_rate = 0.001`, 50 epochs) reduces to item bias. For 6,793 of 7,622 personalized users, its top 10 equals the B0 top 10. Tune B2 on the `t0` validation split before using it as a baseline, and record the tuning budget. Do not tune on these test results.
