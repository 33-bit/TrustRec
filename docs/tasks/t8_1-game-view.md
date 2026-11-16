# Task: T8.1: Reselect the explained recommender on the game-only view

Status: `t0` tuning done; `t1` results below. Decision:
[ADR-0022](../adr/0022-game-only-recommendation-view.md).

## Game-only view

```bash
PYTHONPATH=src python3 scripts/classify_item_scope.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cutoff t0 --curated artifacts/curated/gpt-6-luna --output artifacts/scope/item_scope_t0.json
PYTHONPATH=src python3 scripts/build_game_view.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --t0-scope artifacts/scope/item_scope_t0.json --t1-scope artifacts/scope/item_scope_t1.json --output-dir data/processed/video_games/video_games-full-d6c4efeb74aa-games
```

| | `t0` scope | `t1` scope |
| --- | --- | --- |
| Store credit | 222 | 222 |
| Hardware or accessories | 903 | 1,015 |

| Table | Rows | Game rows |
| --- | --- | --- |
| `train_interactions` | 1,373,677 | 1,283,111 |
| `validation_targets` | 95,559 | 81,448 |
| `fit_interactions` | 1,545,387 | 1,434,666 |
| `test_targets` | 76,435 | 60,249 |

The view's snapshot ID is `video_games-full-d6c4efeb74aa-games`, and its
dataset hash is
`e38af6bdcdcc228717b0a884d65939eea12413886c7b83ec0d1926d064c76021`.

## Tuning on `t0` (63,239 validation users)

Popularity (B0) scores NDCG@10 0.0144 on validation.

B5 window, 9 trials. Tuning report SHA-256:
`e1be6e5077ba189afdedf68e1ed6b56daac60ae4a0d56e9108531b7304123943`.

| Window (days) | 30 | 60 | 90 | 180 | 365 | 540 | 730 | 1,095 | 1,825 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NDCG@10 | **0.0622** | 0.0612 | 0.0620 | 0.0611 | 0.0442 | 0.0365 | 0.0276 | 0.0210 | 0.0144 |

B5 with 30 days beats B0, so it is the cold-start fallback and the recency
term (ADR-0022, rule 3).

H0 and T0 use the `adr0022` grid, 9 trials each. Tuning report SHA-256:
`4d664769b0da4c3d68b52a6c8934d570e9a1f1790f3e8cb72c4f8c8ee3bd5212`.
Scoring took 11 minutes on 4 cores.

| Model | Range | Selected setting | NDCG@10 |
| --- | --- | --- | --- |
| H0 | 0.0573 to 0.0611 | MF 0, graph 0.4, recency 0.4, aspect 0.2 | **0.0611** |
| T0 | 0.0573 to 0.0599 | `g_max` 0.1, `tau` 0.4, `rho` 0.2 | 0.0599 |

- **Recency drives the score.** Scores rise with the recency weight (0, 0.1,
  0.2, and 0.4 give about 0.0574, 0.0578, 0.0586, and 0.0611). The grid stops
  at 0.4, so the best H0 sits at its edge.
- **The other weights barely matter.** MF weight, aspect weight, and `g_max`
  each change NDCG@10 by less than 0.0002.
- **Selection.** H0 is selected as the explained recommender (ADR-0022, rule
  4). On validation it is 0.0011 below the strongest baseline, B5.
