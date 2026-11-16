# Task: T8.1: Reselect the explained recommender on the game-only view

Status: done. `t0` tuning, `t1` test, explanations, verification, and an independent audit. Decision:
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

## Test on `t1` (47,307 users with a positive game target)

The run is `t8-games-t1-models`, metrics SHA-256
`f5526e6104df03d94716b026bb57d7ae507dc56e907113859d2aff4ab2cc2f56`. The
evaluator reports no exclusions. Every model ranks game items only. The
reference is B5 (30 days), the strongest baseline on `t0`.

| Model | NDCG@10 | Difference from B5 | Recall@10 |
| --- | --- | --- | --- |
| **H0, selected** (graph 0.4, recency 0.4, aspect 0.2) | **0.0238** | +0.0009 [+0.0005, +0.0013] | 0.0515 |
| T0 (`g_max` 0.1, `tau` 0.4) | 0.0237 | +0.0009 [+0.0005, +0.0012] | 0.0513 |
| H0 without aspect (graph 0.5, recency 0.5) | 0.0237 | +0.0008 [+0.0005, +0.0012] | 0.0515 |
| B5 recent popularity, 30 days | 0.0229 | | 0.0504 |
| B3 PPR | 0.0160 | -0.0069 [-0.0081, -0.0057] | 0.0305 |
| B2 BPR MF | 0.0146 | -0.0083 | 0.0285 |
| B0 most popular | 0.0146 | -0.0083 [-0.0095, -0.0072] | 0.0284 |

| Slice | Users | B0 | B3 | B5 | H0 | H0 without aspect |
| --- | --- | --- | --- | --- | --- | --- |
| no_history | 40,459 | 0.0150 | 0.0150 | 0.0232 | 0.0232 | 0.0232 |
| 1-2 | 4,352 | 0.0121 | 0.0235 | 0.0208 | 0.0285 | 0.0277 |
| 3-5 | 1,309 | 0.0146 | 0.0223 | 0.0238 | 0.0279 | 0.0269 |
| >5 | 1,187 | 0.0102 | 0.0170 | 0.0182 | 0.0218 | 0.0225 |

The paired user bootstrap of H0 minus H0 without aspect, computed from the
per-user rows with 2,000 samples and seed 7, is +0.00008 NDCG@10
[-0.00006, +0.00022].

Findings:

- **The guardrail holds.** On the game-only view, the selected H0 is better
  than B5 (the interval is above 0) and on par with it (the interval lies
  within ±0.005). This is the ADR-0018 guardrail against the strongest
  baseline.
- **Recency is a strong game signal too.** Once store credit is gone, B5
  still beats B0 by about 57% and B3 by about 43%. B3 PPR beats B0 only for
  users with history.
- **The hybrid gains on users with history.** H0 beats both B5 and B3 for
  users with 1 to 5 positive interactions, by blending graph and recency.
- **Aspect evidence adds no measurable accuracy** on this view either. The
  explained recommender is therefore chosen for its accuracy, and its aspect
  term is what lets it carry explanations.

## Explanations for the selected H0 (`t1`, top-10)

```bash
PYTHONPATH=src python3 scripts/run_explanations_batch.py --snapshot-manifest data/processed/video_games/video_games-full-d6c4efeb74aa-games/snapshot.json --evidence artifacts/evidence/t1_gpt-6-luna.jsonl --ranking artifacts/rankings/games/h0_t1.jsonl --h0 '{"mf_weight": 0.0, "graph_weight": 0.4, "recency_weight": 0.4, "aspect_weight": 0.2}' --mf-model artifacts/models/games/b2_t1.npz --graph-params '{"damping": 0.85, "max_iter": 100}' --normalization minmax --cold-start fallback --cold-window-days 30 --recency-window-days 30 --k 10 --max-claims 1 --workers 4 --output-dir artifacts/explanations/games_t1_h0_k10
```

Summary SHA-256:
`17197fcc6e9a5a77eec29d65ec73a8a9ad0d04bce584ad5073bda5f257f6f937`. The
ranking check and the profile check pass. The run explains 4,951 distinct
items. Before verification, 99.6% of recommendations carry a supported claim.

| | Full catalog (T7.1, second run) | Game-only view |
| --- | --- | --- |
| Supported claims: positive / mixed / negative | 89.0% / 10.6% / 0.3% | 61.5% / 29.0% / 9.5% |
| Share about gameplay | 96.9% | 69.0% |
| Next aspects | performance 1.1%, value 0.7% | content and replay 10.1%, value 9.5%, graphics 9.3% |
| Faithful (aspect in score) | 98.3% of 74,364 | 98.2% of 60,607 |
| Leaves top-10 after cited / random removal | 14.1% / 0.9% | 14.4% / 1.3% |

Without store credit, explanations are more varied. More aspects appear, and
there are more trade-off (negative) and mixed claims. The faithfulness result
is unchanged.

### Verified display passages (ADR-0020)

Verification of all 6,843 distinct displayed claims finished after a
DeepSeek top-up. The first run stopped at 3,593 claims when the balance ran
out; the rerun resumed from the cache with 3,250 more requests (1.71M input
and 0.93M output tokens). Verification summary SHA-256:
`cea989e0e66414ecd7119da8f84c5c130926312cd1a0748744cfd8eb05076a91`.

| | Value |
| --- | --- |
| Displayed claims kept | 6,649 of 6,843 (97.2%) |
| Views kept | 470,832 of 471,200 (99.9%) |
| Recommendations with a supported, verified claim | 99.5% |
| Where the aspect is in the score | 97.0% of 62,120 |
| Cold-start (descriptive only) | 99.9% of 410,950 |
| Supported claims: positive / mixed / negative | 61.5% / 29.0% / 9.5% |
| Aspects | gameplay 69.1%, content and replay 10.1%, value 9.6%, graphics 9.3%, others 1.9% |

### Independent audit (`deepseek-v4-pro`)

The judge is a different model from the verifier, with reasoning off,
temperature 0, and prompt `t7.3-claim-judge-v1`. It judged 100 claims per
direction plus the 30 most-shown claims: 330 requests, 135k input and 39k
output tokens, no failures. Audit SHA-256:
`6218831a154e567e2e09d116d891ce7de0877be32ac25bb537a2800fc8376865`.

| Stratum | Aspect correct | About the product | Polarity agrees with label | Derived support |
| --- | --- | --- | --- | --- |
| Positive | 99.6% | 100% | 96.8% | 93.0% |
| Negative | 96.8% | 100% | 94.7% | 91.0% |
| Mixed | 98.6% | 100% | 93.6% | 93.0% |
| Top 30 by exposure | 97.7% | 100% | 96.6% | 96.7% (90.1% weighted by exposure) |

All 950 cited passages trace to their source text.

### Findings

- **The full TrustRec pipeline holds on games.** The explained recommender
  beats B5 on the game-only view (+0.0009 NDCG@10). 99.5% of its top-10
  recommendations carry a verified claim. An independent judge supports
  92.3% of sampled claims and 90.1% of exposure-weighted top claims.
- **Faithfulness and its limit.** Where the aspect term is in the score,
  98.2% of claims are faithful. For the 87% of recommendations that go to
  users without history, the claim describes the item but does not explain
  why it was recommended. Reports must say so (ADR-0010).
- **Accuracy contribution of aspects.** Aspect evidence still adds no
  measurable accuracy (+0.00008, interval includes 0). Its value in this
  system is explanation, not ranking.
