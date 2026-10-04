# T3.1: Popularity and Item kNN

Status: Done.

This task implements the B0 most-popular baseline and the B1 item kNN
baseline. Both models use the same candidate contract.

## Related records

- Backlog: [`backlog.md`](backlog.md)
- Temporal protocol: [`ADR-0002`](../adr/0002-temporal-snapshot-protocol.md)
- Score contract: [`ADR-0009`](../adr/0009-score-normalization-and-gating.md)
- Ranking contract: [`ranking-contract.md`](../specs/ranking-contract.md)
- Recommendation protocol: [`recommendation-protocol.md`](../specs/recommendation-protocol.md)
- Phase plan: [`phase-3-ranking.md`](../plans/phase-3-ranking.md)
- Design: [`2026-10-04-t3-1-ranking-baselines-design.md`](../superpowers/specs/2026-10-04-t3-1-ranking-baselines-design.md)
- Implementation plan: [`2026-10-04-t3-1-ranking-baselines.md`](../superpowers/plans/2026-10-04-t3-1-ranking-baselines.md)

## Source and cutoff

The source snapshot is `video_games-full-d6c4efeb74aa`. Its dataset hash is
`0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`.
The training cutoff is `2019-01-13T04:24:39.377000+00:00`. The final fit
cutoff is `2020-08-31T22:46:09.247000+00:00`.

The rankers use the first event for each user-item pair with a timestamp
before the supplied cutoff. A rating of 4 or higher is positive. A missing
rating is unknown. A user history removes all reviewed items from the
candidate set, including low-rated items.

## Outputs

The implementation uses these files:

- `src/trustrec/recommenders/contracts.py`
- `src/trustrec/recommenders/popularity.py`
- `src/trustrec/recommenders/item_knn.py`
- `scripts/run_ranking_baselines.py`

`CandidateSet` contains the user ID, snapshot ID, cutoff, history IDs, and
candidate IDs. `RankingResult` contains ranked items, component scores,
configuration, seed, model hash, and a fallback reason.

The popularity model counts positive first events. Item kNN builds binary
item-to-user sets and sums cosine similarity to positive history items.
Item kNN falls back to positive popularity when the user has no positive
history or no similar candidate. Empty candidate sets return no ranked items.

The reproducible command is `scripts/run_ranking_baselines.py`. It reads
`train_interactions` for `t0` or `fit_interactions` for `t1`. It writes
a JSON artifact with snapshot, dataset, cutoff, configuration, and model
hashes.

## Reproduce the checks

Run these commands from the repository root:

```bash
ruff format src/trustrec/recommenders tests/unit/test_recommender_contracts.py tests/unit/test_recommender_popularity.py tests/unit/test_recommender_item_knn.py
ruff check src/trustrec/recommenders tests/unit/test_recommender_contracts.py tests/unit/test_recommender_popularity.py tests/unit/test_recommender_item_knn.py
python3 -m pytest tests/unit/test_recommender_contracts.py tests/unit/test_recommender_popularity.py tests/unit/test_recommender_item_knn.py -q
python3 -m pytest tests/integration/test_ranking_baselines_script.py -q
```

The test fixtures use fixed IDs, timestamps, ratings, and no downloaded
source data. The model hash covers the positive threshold, cutoff, model
configuration, and fitted counts or item vectors.
