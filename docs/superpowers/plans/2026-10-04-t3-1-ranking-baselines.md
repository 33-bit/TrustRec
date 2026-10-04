# T3.1 Ranking Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic most-popular and item kNN recommenders that use one shared, time-safe candidate contract.

**Architecture:** `recommenders/contracts.py` defines validated candidate and ranking result records plus interaction coercion. `popularity.py` and `item_knn.py` fit on first user-item events before a cutoff and rank the exact candidate IDs supplied by `CandidateSet`. `scripts/run_ranking_baselines.py` reads snapshot Parquet and writes a lineage-bearing JSON artifact. The rankers return score parts, configuration, seed, and a deterministic model hash.

**Tech Stack:** Python 3.11, dataclasses, standard-library hashing and math, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-04-t3-1-ranking-baselines-design.md`

## Global Constraints

- Use events before the cutoff for all ranking features.
- Treat rating 4 or higher as a positive ranking signal.
- Keep missing feedback as unknown, not negative.
- Use identical candidate IDs for compared models.
- Sort ties by ascending item ID.
- Keep generated data and raw source files outside Git.

---

### Task 1: Shared ranking contract

**Files:**
- Create: `src/trustrec/recommenders/contracts.py`
- Modify: `src/trustrec/recommenders/__init__.py`
- Test: `tests/unit/test_recommender_contracts.py`

**Interfaces:**
- Produces `CandidateSet`, `RankedItem`, `RankingResult`, `build_candidate_set`, and `coerce_interactions`.
- Rankers consume `CandidateSet` and return `RankingResult`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_candidate_set_removes_all_pre_cutoff_history_items():
    result = build_candidate_set(
        "u1",
        "snap-1",
        [
            {"user_id": "u1", "item_id": "seen-low", "rating": 2, "timestamp": 1},
            {"user_id": "u1", "item_id": "seen-high", "rating": 5, "timestamp": 2},
            {"user_id": "u2", "item_id": "candidate", "rating": 5, "timestamp": 3},
            {"user_id": "u1", "item_id": "future", "rating": 5, "timestamp": 20},
        ],
        cutoff_timestamp=10,
    )
    assert result.candidate_item_ids == ("candidate",)
    assert result.history_item_ids == frozenset({"seen-low", "seen-high"})


def test_candidate_set_rejects_history_overlap():
    with pytest.raises(ValueError, match="exclude history"):
        CandidateSet(
            user_id="u1",
            snapshot_id="snap-1",
            cutoff_timestamp=10,
            candidate_item_ids=("i1",),
            history_item_ids=frozenset({"i1"}),
        )


def test_ranking_result_keeps_ranked_items_and_candidate_ids():
    item = RankedItem("i1", 1, 2.0, {"popularity": 2.0})
    result = RankingResult(
        model_id="b0_most_popular",
        user_id="u1",
        snapshot_id="snap-1",
        cutoff_timestamp=10,
        candidate_item_ids=("i1",),
        items=(item,),
        configuration={"positive_threshold": 4.0},
        seed=None,
        model_hash="a" * 64,
    )
    assert result.items[0].component_scores["popularity"] == 2.0
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `python3 -m pytest tests/unit/test_recommender_contracts.py -q`

Expected: collection fails because `trustrec.recommenders.contracts` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Add frozen dataclasses with non-empty ID validation, unique candidate validation,
history exclusion validation, non-negative ranks, and finite scores. Convert
iterables and pandas-like objects through `to_dict("records")`, then require
`user_id`, `item_id`, and `timestamp` fields.

- [ ] **Step 4: Run the tests to verify that they pass**

Run: `python3 -m pytest tests/unit/test_recommender_contracts.py -q`

Expected: all contract tests pass.

- [ ] **Step 5: Export the public names**

Export the contract types and helper functions from
`src/trustrec/recommenders/__init__.py`.

### Task 2: Most-popular ranker

**Files:**
- Create: `src/trustrec/recommenders/popularity.py`
- Test: `tests/unit/test_recommender_popularity.py`
- Modify: `configs/experiments/b0_most_popular.toml`

**Interfaces:**
- Consumes `CandidateSet` and interaction rows.
- Produces `PopularityRanker.fit(...).rank(candidate_set, k=None) -> RankingResult`.
- Produces `rank_popularity(interactions, candidate_set, ...) -> RankingResult`.

- [ ] **Step 1: Write the failing tests**

```python
def test_popularity_uses_positive_pre_cutoff_counts_and_stable_ties():
    result = rank_popularity(
        [
            {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u2", "item_id": "a", "rating": 4, "timestamp": 2},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 3},
            {"user_id": "u4", "item_id": "b", "rating": 3, "timestamp": 4},
            {"user_id": "u5", "item_id": "c", "rating": 5, "timestamp": 11},
        ],
        CandidateSet("u9", "snap-1", 10, ("a", "b", "c")),
    )
    assert [item.item_id for item in result.items] == ["a", "b", "c"]
    assert [item.total_score for item in result.items] == [2.0, 1.0, 0.0]


def test_popularity_returns_empty_result_for_empty_candidates():
    result = rank_popularity([], CandidateSet("u1", "snap-1", 10, ()))
    assert result.items == ()
    assert result.fallback_reason == "no_candidates"
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `python3 -m pytest tests/unit/test_recommender_popularity.py -q`

Expected: collection fails because `trustrec.recommenders.popularity` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Keep the first event for each user-item pair with `timestamp <
cutoff_timestamp`. Count each first event with `rating >= positive_threshold`
by candidate item.
Create one `RankedItem` for each candidate, sort by `(-score, item_id)`, and
assign one-based ranks after applying `k`.

- [ ] **Step 4: Run the tests to verify that they pass**

Run: `python3 -m pytest tests/unit/test_recommender_popularity.py -q`

Expected: all popularity tests pass.

- [ ] **Step 5: Add the locked configuration fields**

Add `positive_threshold = 4` and `tie_break = "item_id_ascending"` to
`configs/experiments/b0_most_popular.toml`.

### Task 3: Item kNN ranker

**Files:**
- Create: `src/trustrec/recommenders/item_knn.py`
- Test: `tests/unit/test_recommender_item_knn.py`
- Modify: `configs/experiments/b1_item_knn.toml`

**Interfaces:**
- Consumes the same `CandidateSet` and interaction rows as popularity.
- Produces `ItemKNNRanker.fit(...).rank(candidate_set, k=None) -> RankingResult`.
- Produces `rank_item_knn(interactions, candidate_set, ...) -> RankingResult`.

- [ ] **Step 1: Write the failing tests**

```python
def test_item_knn_scores_candidates_by_cosine_similarity():
    result = rank_item_knn(
        [
            {"user_id": "u1", "item_id": "h", "rating": 5, "timestamp": 1},
            {"user_id": "u2", "item_id": "h", "rating": 5, "timestamp": 2},
            {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 3},
            {"user_id": "u3", "item_id": "a", "rating": 5, "timestamp": 4},
            {"user_id": "u2", "item_id": "b", "rating": 5, "timestamp": 5},
            {"user_id": "u4", "item_id": "b", "rating": 5, "timestamp": 6},
        ],
        CandidateSet("u1", "snap-1", 10, ("b",), frozenset({"h"})),
    )
    assert result.items[0].item_id == "b"
    assert result.items[0].component_scores["knn"] == pytest.approx(2**-0.5)


def test_item_knn_falls_back_to_popularity_without_positive_history():
    result = rank_item_knn(
        [
            {"user_id": "u2", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u3", "item_id": "b", "rating": 5, "timestamp": 2},
        ],
        CandidateSet("u1", "snap-1", 10, ("a", "b")),
    )
    assert result.fallback_reason == "no_positive_history"
    assert [item.item_id for item in result.items] == ["a", "b"]
```

- [ ] **Step 2: Run the tests to verify that they fail**

Run: `python3 -m pytest tests/unit/test_recommender_item_knn.py -q`

Expected: collection fails because `trustrec.recommenders.item_knn` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Build item-to-user sets from first positive rows before the cutoff. Compute cosine
similarity as shared users divided by the square root of the two item
degrees. Sum each candidate's similarities to the user's positive history,
optionally keeping only its strongest `n_neighbors` similarities. If the
history is empty or all similarities are zero, sort by the stored popularity
count and record `no_positive_history` or `no_similar_items`.

- [ ] **Step 4: Run the tests to verify that they pass**

Run: `python3 -m pytest tests/unit/test_recommender_item_knn.py -q`

Expected: all item kNN tests pass.

- [ ] **Step 5: Add the locked configuration fields**

Add `positive_threshold = 4`, `tie_break = "item_id_ascending"`, and
`neighbor_limit = 0` to `configs/experiments/b1_item_knn.toml`. Interpret
zero as unlimited neighbors.

### Task 4: Task record and repository verification

**Files:**
- Create: `docs/tasks/t3_1-ranking-baselines.md`
- Create: `scripts/run_ranking_baselines.py`
- Test: `tests/integration/test_ranking_baselines_script.py`
- Modify: `docs/tasks/backlog.md`
- Modify: `src/trustrec/recommenders/README.md`

**Interfaces:**
- Documents the snapshot, cutoff, model, output, and validation contract.
- Records the T3.1 status and output paths in the backlog.

- [ ] **Step 1: Write the task record**

Record the source snapshot ID `video_games-full-d6c4efeb74aa`, the
pre-cutoff rule, the positive threshold, the output module paths, the test
commands, and the fallback cases.

- [ ] **Step 2: Update the backlog**

Change T3.1 to `done` and add the output path
`src/trustrec/recommenders/{contracts,popularity,item_knn}.py`.

- [ ] **Step 3: Run focused and full checks**

Run:

```bash
python3 -m pytest tests/unit/test_recommender_contracts.py tests/unit/test_recommender_popularity.py tests/unit/test_recommender_item_knn.py -q
python3 -m pytest tests/integration/test_ranking_baselines_script.py -q
ruff check src/trustrec/recommenders tests/unit/test_recommender_contracts.py tests/unit/test_recommender_popularity.py tests/unit/test_recommender_item_knn.py
python3 -m pytest -q
```

Expected: all focused tests, Ruff checks, and the full suite pass with zero
failures.
