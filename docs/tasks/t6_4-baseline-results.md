# Task: T6.4 (partial): Full Video Games baseline results

Status: B0 through B3, H0, and T0 are done on the real `t1` test set. Every
tuned model received a 9-trial validation budget on `t0`. B2 is run with seeds
7, 17, and 27; H0 and T0 use the seed-7 BPR model. See the H0 and T0 update at
the end.

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

- On this very sparse data (about 1.7 interactions per user), BPR does not beat popularity. A wider BPR search, such as fewer dimensions or more negative samples, would need a new recorded budget.

## B1 and B3 tuning on `t0` (update)

`scripts/tune_baseline.py` gives B1 and B3 the same 9-trial validation budget
as B2. It uses 75,414 validation users, seed 7, and 200 bootstrap samples.
Validation popularity NDCG@10 is 0.0378.

| Model | Grid | Best setting | Best NDCG@10 | Difference from popularity |
| --- | --- | --- | --- | --- |
| B1 item kNN | `neighbor_limit` in {all, 5, 10, 20, 50, 100, 200, 500, 1000} | all neighbors | 0.0356 | -0.00225 [-0.00262, -0.00194] |
| B3 PPR | damping {0.5, 0.7, 0.85} × `max_iter` {10, 30, 100} | 0.85, 100 | 0.0392 | +0.00135 [+0.00099, +0.00166] |

Every B1 setting scores about the same and stays below popularity. B3
improves as damping rises. Both winners equal the default settings, so the
B1 and B3 test runs above are already the tuned runs. The B1 grid took 446 s
and the B3 grid took 3,323 s on 4 cores.

| Model | Validation tuning | Test NDCG@10 | Difference from B0 on test |
| --- | --- | --- | --- |
| B0 most popular | none | 0.0268 | |
| B1 item kNN | 9 trials | 0.0256 | -0.0012 [-0.0016, -0.0008] |
| B2 BPR MF, mean of 3 seeds | 9 trials | 0.0268 | about 0 (seed SD 0.0004) |
| B3 PPR | 9 trials | 0.0284 | +0.0015 [+0.0012, +0.0020] |

## H0 and T0 with gpt-6-luna evidence (update, 2026-11-04)

T6.3 ran H0 (fixed hybrid) and T0 (TrustRec gate) with the curated
gpt-6-luna labels ([LLM label dataset](../specs/llm-label-dataset.md)). The
`t0` evidence has 3,278,623 rows from 1,373,677 reviews; the `t1` evidence has
3,490,130 rows. Components are the tuned BPR MF (`b2_t0_best.npz` on `t0`,
`b2_tuned_t1_seed07.npz` on `t1`) and PPR with damping 0.85 and 100
iterations. Code revision: `b112306`.

```bash
PYTHONPATH=src python3 scripts/label_corpus_evidence.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cutoff t0 --workers 1 --labeler llm-cache --label-cache artifacts/curated/gpt-6-luna --output artifacts/evidence/t0_gpt-6-luna.jsonl
PYTHONPATH=src python3 scripts/run_hybrid_batch.py tune --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --evidence artifacts/evidence/t0_gpt-6-luna.jsonl --mf-params '{"learning_rate": 0.01, "regularization": 0.0001, "epochs": 30, "seed": 7}' --mf-model artifacts/models/b2_t0_best.npz --graph-params '{"damping": 0.85, "max_iter": 100}' --workers 4 --output reports/generated/t6-3-hybrid-tuning-t0-gpt-6-luna.json
PYTHONPATH=src python3 scripts/run_hybrid_batch.py run --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cutoff t1 --evidence artifacts/evidence/t1_gpt-6-luna.jsonl --mf-model artifacts/models/b2_tuned_t1_seed07.npz --graph-params '{"damping": 0.85, "max_iter": 100}' --workers 4 --k 100 --h0 '{"mf_weight": 0.2, "graph_weight": 0.6, "aspect_weight": 0.2}' --h0-output artifacts/rankings/h0_t1_gpt-6-luna.jsonl --t0 '{"kappa": 5.0, "g_max": 0.25, "rho": 0.2}' --t0-output artifacts/rankings/t0_t1_gpt-6-luna.jsonl
```

### Validation tuning on `t0`

Each model had a 9-trial budget, selected by validation NDCG@10 on 75,414
users. Validation popularity NDCG@10 is 0.0378. Scoring all 18 trials took
27 minutes on 4 cores. Tuning report SHA-256:
`9cc21749b5db23d6bf8aca6255eb205414cc56d4aff79369deaf0f213d465682`.

| Model | Grid | Best setting | Best NDCG@10 | Difference from popularity |
| --- | --- | --- | --- | --- |
| H0 | 9 weight triples | MF 0.2, graph 0.6, aspect 0.2 | 0.0330 | -0.0048 [-0.0056, -0.0040] |
| T0 | `g_max` {0.25, 0.5, 0.75} × `rho` {0.2, 0.5, 0.8}, `kappa` 5 | `g_max` 0.25, `rho` 0.2 | 0.0324 | -0.0054 [-0.0063, -0.0047] |

Every trial is below popularity on validation. Scores fall as the aspect
weight or `g_max` rises: aspect weight 0.2 gives about 0.033 and 0.4 gives
about 0.022; `g_max` 0.75 gives about 0.021. `rho` barely matters, because MF
and graph both fall back to popularity for users without history.

### Test results on `t1`

Same 61,775 users, candidate set hash, and bootstrap settings as the baseline
table. Metrics SHA-256:
`cc32066ef8f512124527e0171188ee623a032e2728f907af9b079af57dce4b7a`
(run `t6-3-video-games-t1-hybrids-gpt-6-luna`).

| Model | NDCG@10 | Difference from B0 | Recall@10 | Difference from B0 |
| --- | --- | --- | --- | --- |
| B0 most popular | 0.0268 | | 0.0492 | |
| B3 PPR | 0.0284 | +0.0015 [+0.0012, +0.0020] | 0.0512 | +0.0021 |
| H0 fixed hybrid | 0.0384 | +0.0116 [+0.0107, +0.0125] | 0.0575 | +0.0083 [+0.0070, +0.0096] |
| T0 TrustRec | 0.0362 | +0.0094 [+0.0085, +0.0104] | 0.0506 | +0.0014 [-0.0002, +0.0029] |

| Slice | Users | B0 | B3 | H0 | T0 |
| --- | --- | --- | --- | --- | --- |
| no_history | 53,413 | 0.0276 | 0.0276 | 0.0404 | 0.0376 |
| 1-2 | 5,464 | 0.0237 | 0.0371 | 0.0284 | 0.0302 |
| 3-5 | 1,546 | 0.0201 | 0.0294 | 0.0255 | 0.0277 |
| >5 | 1,352 | 0.0155 | 0.0215 | 0.0180 | 0.0185 |

### Findings

- The validation and test results disagree in sign. The selection followed the
  protocol (validation only), so the test numbers stand, but they are not
  evidence that the hybrid personalizes better.
- The whole test gain comes from users without positive history. For them, H0
  is a non-personalized list: 53,636 of 54,153 such users get the same top 10.
  That list keeps popular items with good aspect evidence and adds two items
  released in 2020 (`B087NN2K41`, `B087SHFL9B`). It drops items that are popular
  only by all-time count. All-time popularity is a stale signal at `T1`, and
  percentile normalization plus the aspect term breaks its dominance.
- No leakage was found. Evidence rows are checked to be strictly before each
  cutoff, popularity uses only `fit_interactions`, and the protocol reports no
  exclusions.
- For users with history, B3 PPR stays best. H0 and T0 are below B3 in every
  history slice, because they dilute the graph score with aspect percentiles.
- T0's gate stays small (mean 0.056 on `t1`), but it still reorders the top of
  the popularity list. Among the most popular items, percentile scores differ by
  about 1e-5, so a small aspect term is enough to change the top 10.

Open points:

- A recency-aware popularity baseline would separate "fresher popular items"
  from "aspect evidence". It needs a new baseline row and budget.
- Using no aspect term for users without history, or a score normalization
  that keeps popularity magnitude, changes ADR-0009 and needs an owner
  decision before any run.

## B5 recent popularity (update, 2026-11-05)

[ADR-0016](../adr/0016-recent-popularity-baseline.md) adds B5, which ranks by
positive interactions in a window before the cutoff. It tests whether the H0
and T0 test gain came from recency instead of aspect evidence.

```bash
PYTHONPATH=src python3 scripts/tune_baseline.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b5_recent_popularity --workers 4 --output reports/generated/t6-5-b5-tuning-t0.json
PYTHONPATH=src python3 scripts/run_rankings_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --model b5_recent_popularity --cutoff t1 --k 100 --workers 4 --window-days 60 --output artifacts/rankings/b5_t1_w60.jsonl
```

Validation tuning on `t0` (9 trials, 75,414 users, popularity 0.0378). Tuning
report SHA-256: `d42f766719779dd9ce7d12bfbfb4b23e30a363d7e5d2087495a8cefdb899bd93`.

| Window (days) | 30 | 60 | 90 | 180 | 365 | 540 | 730 | 1,095 | 1,825 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NDCG@10 | 0.0769 | **0.0775** | 0.0773 | 0.0772 | 0.0667 | 0.0627 | 0.0570 | 0.0497 | 0.0408 |

The selected window is 60 days (+0.0396 [+0.0384, +0.0410] over popularity).

Test on `t1`, with B5 as the reference. Run `t6-5-video-games-t1-with-b5`,
metrics SHA-256 `5f20a018b978b42211326da7d7d32be9f83ffa1706b24a12991a226affc02689`.

| Model | NDCG@10 | Difference from B5 | Recall@10 |
| --- | --- | --- | --- |
| B5 recent popularity, 60 days | 0.1064 [0.1042, 0.1086] | | 0.1557 |
| H0 fixed hybrid | 0.0384 | -0.0680 [-0.0703, -0.0657] | 0.0575 |
| T0 TrustRec | 0.0362 | -0.0702 [-0.0725, -0.0680] | 0.0506 |
| B3 PPR | 0.0284 | -0.0780 [-0.0803, -0.0759] | 0.0512 |
| B0 most popular | 0.0268 | -0.0796 [-0.0818, -0.0774] | 0.0492 |

| Slice | Users | B0 | B3 | B5 | H0 | T0 |
| --- | --- | --- | --- | --- | --- | --- |
| no_history | 53,413 | 0.0276 | 0.0276 | 0.1139 | 0.0404 | 0.0376 |
| 1-2 | 5,464 | 0.0237 | 0.0371 | 0.0674 | 0.0284 | 0.0302 |
| 3-5 | 1,546 | 0.0201 | 0.0294 | 0.0459 | 0.0255 | 0.0277 |
| >5 | 1,352 | 0.0155 | 0.0215 | 0.0398 | 0.0180 | 0.0185 |

Findings:

- Recency dominates this data. B5 is about four times B0 on test and beats
  every other model in every history slice, including users with more than
  five positive interactions.
- The H0 and T0 gain over B0 is a weak, accidental form of recency. Both stay
  far below B5, so aspect evidence adds nothing measurable here yet.
- Every personalized baseline (B1, B2, B3) and both hybrids fall back to
  all-time popularity, and B3 also ranks warm users by long-run graph mass.
  None of them sees recency. A fair test of aspect evidence needs a recency
  term in the base score, which changes ADR-0009 and the B1–B3 definitions and
  needs an owner decision ([ADR-0017](../adr/0017-hybrid-cold-start-and-normalization.md) is the first proposal).

## H0 and T0 under ADR-0017 (update, 2026-11-05)

[ADR-0017](../adr/0017-hybrid-cold-start-and-normalization.md) (accepted by
the owner) adds three changes to H0 and T0:

- a recency component `R`, the B5 score with a 60-day window;
- a B5 fallback for users without positive history;
- min-max normalization.

The evidence and components match the ADR-0009 runs.

```bash
PYTHONPATH=src python3 scripts/run_hybrid_batch.py tune --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --evidence artifacts/evidence/t0_gpt-6-luna.jsonl --mf-model artifacts/models/b2_t0_best.npz --graph-params '{"damping": 0.85, "max_iter": 100}' --workers 4 --grid adr0017 --normalization minmax --cold-start fallback --cold-window-days 60 --recency-window-days 60 --output reports/generated/t6-5-hybrid-tuning-t0-adr0017.json
```

The `t1` run used `run_hybrids` with the same settings, the selected H0 and
T0, and T0 with `g_max` 0 as the no-aspect ablation.

### Validation tuning on `t0`

9 trials per model, 75,414 users; B5 (60 days) scores 0.0775. Tuning report
SHA-256: `ef6095a8092a0c7dbad4c29dd46f4ef5895bdc8aef6a90912242ede5c2923b20`.
Scoring took 12 minutes on 4 cores.

| Model | Range over 9 trials | Selected setting | NDCG@10 |
| --- | --- | --- | --- |
| H0 | 0.0770 to 0.0777 | MF 0, graph 0.2, recency 0.6, aspect 0.2 | 0.0777 |
| T0 | 0.0760 to 0.0775 | `g_max` 0.1, `tau` 0.8, `rho` 0.5 | 0.0775 |

Every trial is within 0.0015 of B5. Higher recency weight is better. Higher
`g_max` is slightly worse at every `tau`.

### Test on `t1`

Run `t6-5-video-games-t1-adr0017`, reference B5. Metrics SHA-256:
`3b47c037c1ff459a8dd0a48c6b9ee512e8938811eac4bc995c7edaccbd1234d0`.

| Model | NDCG@10 | Difference from B5 | Recall@10 | Difference from B5 |
| --- | --- | --- | --- | --- |
| B5 recent popularity | 0.1064 | | 0.1557 | |
| H0, ADR-0017 | 0.1071 | +0.0007 [+0.0005, +0.0009] | 0.1574 | +0.0016 [+0.0012, +0.0021] |
| T0, ADR-0017 | 0.1068 | +0.0004 [+0.0003, +0.0005] | 0.1569 | +0.0012 [+0.0009, +0.0015] |
| T0 base, no aspect (`g_max` 0) | 0.1068 | +0.0004 [+0.0003, +0.0005] | 0.1569 | +0.0012 [+0.0009, +0.0015] |
| H0, ADR-0009 | 0.0384 | -0.0680 | 0.0575 | |
| T0, ADR-0009 | 0.0362 | -0.0702 | 0.0506 | |

| Slice | Users | B5 | H0 | T0 | T0 no aspect |
| --- | --- | --- | --- | --- | --- |
| no_history | 53,413 | 0.1139 | 0.1139 | 0.1139 | 0.1139 |
| 1-2 | 5,464 | 0.0674 | 0.0733 | 0.0712 | 0.0712 |
| 3-5 | 1,546 | 0.0459 | 0.0510 | 0.0487 | 0.0483 |
| >5 | 1,352 | 0.0398 | 0.0414 | 0.0402 | 0.0404 |

### Findings

- With a recency term, both hybrids beat B5 on test. The gain is small
  (+0.0004 to +0.0007 NDCG@10) and comes only from users with history, because
  users without history get B5 exactly.
- T0 equals its no-aspect ablation in total and almost exactly by slice. Its
  mean gate on `t1` is 0.002, so the TrustRec gate adds no measurable value
  here. The T0 gain over B5 comes from blending MF and graph scores into the
  recency base, not from aspect evidence.
- H0 is the best model. It keeps a 0.2 aspect weight, but it also differs
  from the T0 base in its graph and recency weights. This run cannot say how
  much of H0's lead comes from the aspect term. Answering that needs an H0
  ablation with the aspect weight set to 0.
- Validation predicted these results: on `t0`, every ADR-0017 setting was
  within 0.0015 of B5, and the H0 winner was the best trial.

### H0 aspect ablation on `t1` (update, 2026-11-05)

The selected ADR-0017 H0 uses MF 0, graph 0.2, recency 0.6, and aspect 0.2.
The ablation sets the aspect weight to 0 and rescales the other weights in
proportion: graph 0.25 and recency 0.75. Nothing was tuned on `t1`. Run
`t6-5-video-games-t1-h0-aspect-ablation`, reference H0 without aspect.
Metrics SHA-256:
`5aeadc67568406fd0cfa4d85f497632c47be2494711d98238ef53310cc3297b7`.

| Model | NDCG@10 | Difference from H0 without aspect | Recall@10 | Difference |
| --- | --- | --- | --- | --- |
| H0 with aspect | 0.1071 | +0.00011 [+0.00001, +0.00022] | 0.1574 | +0.00033 [+0.00007, +0.00061] |
| H0 without aspect | 0.1070 | | 0.1570 | |
| B5 | 0.1064 | -0.00058 [-0.00072, -0.00045] | 0.1557 | -0.00130 [-0.00166, -0.00097] |

| Slice | Users | B5 | H0 without aspect | H0 with aspect |
| --- | --- | --- | --- | --- |
| no_history | 53,413 | 0.1139 | 0.1139 | 0.1139 |
| 1-2 | 5,464 | 0.0674 | 0.0723 | 0.0733 |
| 3-5 | 1,546 | 0.0459 | 0.0495 | 0.0510 |
| >5 | 1,352 | 0.0398 | 0.0422 | 0.0414 |

Findings:

- Most of H0's lead over B5 (+0.0007) comes from blending the graph score into
  recency (+0.0006). Aspect evidence adds about +0.0001 NDCG@10. Its 95%
  interval barely excludes zero, and the Recall@10 interval also excludes zero.
- The aspect term helps users with 1 to 5 positive interactions and slightly
  hurts users with more than 5.
- This is weak evidence. It is one seed, one evidence source, one ablation
  rule (proportional rescaling), and a single paired bootstrap with no
  correction for the other comparisons on the same test set. T0's gate showed
  no aspect effect at all. Report the aspect contribution as small and
  uncertain, not as a confirmed gain.
