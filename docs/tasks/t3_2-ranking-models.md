# T3.2: BPR MF and Personalized PageRank

Status: Done.

This task adds the B2 BPR matrix-factorization baseline and the B3
personalized PageRank baseline. Both models use the T3.1 candidate contract.

The source snapshot is `video_games-full-d6c4efeb74aa`. Its dataset hash is
`0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`.
The training cutoff is `2019-01-13T04:24:39.377000+00:00`. The final fit
cutoff is `2020-08-31T22:46:09.247000+00:00`.

## Source and cutoff

Each model keeps the first event for each user-item pair with a timestamp
before the supplied cutoff. A rating of 4 or higher is positive. Missing and
low ratings are not sampled as negative feedback. Candidate items must be
known before the cutoff and must exclude every item in the user's history.

## B2 BPR matrix factorization

`BPRMFRanker` learns user and item vectors with seeded Bayesian Personalized
Ranking updates. For each positive event, it samples items that are unknown
to that user. The result exposes the MF score, configuration, seed, and model
hash. A user without positive history receives the positive-popularity
fallback with `no_positive_history`.

## B3 personalized PageRank

`PersonalizedPageRankRanker` builds a binary graph from positive user-item
edges. It starts the PageRank walk at the requested user and sends dangling
mass back to that user. The result exposes the PPR score, graph configuration,
seed, and model hash. A user without positive edges uses
`no_positive_history`. A disconnected candidate set uses `no_graph_path` and
the positive-popularity fallback.

## Outputs

The implementation uses these files:

- `src/trustrec/recommenders/bpr_mf.py`
- `src/trustrec/graph/personalized_pagerank.py`
- `scripts/run_ranking_baselines.py`

The runner accepts `b2_bpr_mf` and `b3_ppr` and writes the same lineage fields
as T3.1. BPR uses seed 7 when the runner receives no seed.

## Reproduce the checks

Run these commands from the repository root:

```bash
ruff format --check src/trustrec/recommenders/bpr_mf.py src/trustrec/graph/personalized_pagerank.py scripts/run_ranking_baselines.py tests/unit/test_recommender_bpr_mf.py tests/unit/test_personalized_pagerank.py tests/integration/test_ranking_baselines_script.py
ruff check src/trustrec/recommenders/bpr_mf.py src/trustrec/graph/personalized_pagerank.py src/trustrec/recommenders tests/unit/test_recommender_bpr_mf.py tests/unit/test_personalized_pagerank.py tests/integration/test_ranking_baselines_script.py
python3 -m pytest tests/unit/test_recommender_bpr_mf.py tests/unit/test_personalized_pagerank.py tests/integration/test_ranking_baselines_script.py -q
```
