# Task: T6.4 (partial): Full Video Games baseline results

Status: B0 through B3 are done on the real `t1` test set. BPR MF is tuned on
`t0` and run with seeds 7, 17, and 27. H0 and T0 wait for T6.3. B1 and B3 have
not received a matching tuning budget yet.

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

## BPR MF tuning on `t0` (update)

`scripts/tune_bpr.py` fits BPR MF on `train_interactions` and scores
`validation_targets`, with 75,414 eligible users. The setup takes 51 s. The
budget is 9 trials: learning rate {0.01, 0.05, 0.1} × regularization {1e-4,
1e-3, 1e-2}, with 30 epochs, 64 dimensions, batch size 1024, and seed 7. The
selection metric is validation NDCG@10. Intervals use 200 bootstrap samples.
The whole grid takes 34 minutes.

| Learning rate | Regularization | NDCG@10 | Difference from popularity (0.0378) |
| --- | --- | --- | --- |
| 0.01 | 1e-4 | 0.0376 | -0.0002 [-0.00025, -0.00015] |
| 0.01 | 1e-3 | 0.0376 | -0.0002 [-0.00025, -0.00016] |
| 0.01 | 1e-2 | 0.0376 | -0.0003 [-0.00037, -0.00013] |
| 0.05 | 1e-4 | 0.0376 | -0.0002 [-0.00030, -0.00019] |
| 0.05 | 1e-3 | 0.0375 | -0.0003 [-0.00041, -0.00019] |
| 0.05 | 1e-2 | 0.0361 | -0.0017 [-0.00191, -0.00142] |
| 0.1 | 1e-4 | 0.0375 | -0.0003 [-0.00041, -0.00016] |
| 0.1 | 1e-3 | 0.0372 | -0.0007 [-0.00079, -0.00049] |
| 0.1 | 1e-2 | 0.0366 | -0.0012 [-0.00145, -0.00093] |

The selected setting is learning rate 0.01 with regularization 1e-4. Every
trial personalizes: no personalized user keeps the popularity top 10. Every
trial still scores slightly below popularity on validation.

Test results with the selected setting, refit on `fit_interactions`:

| Model | NDCG@10 | Difference from B0 | Recall@10 |
| --- | --- | --- | --- |
| B2 tuned, seed 7 | 0.0264 [0.0255, 0.0274] | -0.00038 [-0.00047, -0.00030] | 0.0484 |
| B2 tuned, seed 17 | 0.0269 [0.0259, 0.0278] | +0.00002 [+0.00002, +0.00003] | 0.0492 |
| B2 tuned, seed 27 | 0.0272 [0.0262, 0.0281] | +0.00035 [+0.00020, +0.00050] | 0.0485 |
| B2 untuned, seed 7 | 0.0268 [0.0259, 0.0277] | +0.00000 [+0.00000, +0.00000] | 0.0492 |

Across the three seeds, tuned B2 has NDCG@10 0.0268 with a seed standard
deviation of 0.0004. The variation between seeds is larger than the difference
from popularity. B3 PPR stays the strongest baseline.

Evaluation run: `t6-4-video-games-t1-baselines-tuned-bpr`. Its metrics SHA-256
is `dbb0cd254db52dde05f114bfcf7aeadfbd8d6c7a49f0d6232a4a028fbc750816`.
Saved models: `artifacts/models/b2_t0_best.npz` (the tuning winner on `t0`) and
`artifacts/models/b2_tuned_t1_seed{07,17,27}.npz`. Each file is about 400 MB.
Reload a model with `--load-model` on the same snapshot.

```bash
PYTHONPATH=src python3 scripts/tune_bpr.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --output reports/generated/t6-4-bpr-tuning-t0.json --workers 4 --save-best artifacts/models/b2_t0_best.npz
PYTHONPATH=src python3 scripts/run_rankings_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b2_bpr_mf --cutoff t1 --k 100 --workers 4 --seed 7 --learning-rate 0.01 --regularization 0.0001 --epochs 30 --output artifacts/rankings/b2_tuned_t1_seed07.jsonl --save-model artifacts/models/b2_tuned_t1_seed07.npz
```

Open points:

- The protocol asks for an equal tuning budget per model. B1 (`neighbor_limit`) and B3 (`damping`, `max_iter`) should each get 9 validation trials before the final comparison.
- On this very sparse data (about 1.7 interactions per user), BPR does not beat popularity. A wider BPR search, such as fewer dimensions or more negative samples, would need a new recorded budget.
