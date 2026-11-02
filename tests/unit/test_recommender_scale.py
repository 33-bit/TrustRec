import math
import random
import time

import numpy as np
import pyarrow as pa
import pytest

from trustrec.graph import PersonalizedPageRankRanker
from trustrec.recommenders import (
    BPRMFRanker,
    ItemKNNRanker,
    PopularityRanker,
    build_candidate_set,
    cosine_similarity,
)
from trustrec.recommenders.bpr_mf import _sample_negatives
from trustrec.recommenders.matrix import build_interaction_matrix

RANKER_FACTORIES = [
    lambda: PopularityRanker(),
    lambda: ItemKNNRanker(),
    lambda: ItemKNNRanker(n_neighbors=2),
    lambda: BPRMFRanker(epochs=5, embedding_dimensions=8, batch_size=16),
    lambda: PersonalizedPageRankRanker(max_iter=30, batch_size=4),
]


def _rows(seed: int = 11, count: int = 900) -> list[dict[str, object]]:
    generator = random.Random(seed)
    return [
        {
            "review_id": f"r{position}",
            "user_id": f"u{generator.randint(0, 79):02d}",
            "item_id": f"i{int(generator.paretovariate(1.3)) % 40:02d}",
            "rating": float(generator.randint(1, 5)),
            "timestamp": generator.randint(0, 100),
        }
        for position in range(count)
    ]


@pytest.mark.parametrize("factory", RANKER_FACTORIES)
def test_recommend_matches_rank_on_the_protocol_candidates(factory) -> None:
    rows = _rows()
    ranker = factory().fit(rows, 90, snapshot_id="s")
    users = sorted({row["user_id"] for row in rows})
    if hasattr(ranker, "prepare"):
        ranker.prepare(users)
    for user_id in users:
        candidates = build_candidate_set(user_id, "s", rows, 90)
        ranked = ranker.rank(candidates, k=10)
        recommended = ranker.recommend(user_id, 10)

        assert list(recommended.item_ids) == [item.item_id for item in ranked.items]
        assert recommended.fallback_reason == ranked.fallback_reason
        assert recommended.candidate_count == len(candidates.candidate_item_ids)


def test_item_knn_matches_the_set_based_reference() -> None:
    rows = _rows()
    ranker = ItemKNNRanker(n_neighbors=3).fit(rows, 90)
    matrix = ranker.matrix
    item_users: dict[str, set[str]] = {}
    for user_index, user_id in enumerate(matrix.user_ids):
        start, stop = matrix.positive.indptr[user_index], matrix.positive.indptr[user_index + 1]
        for item_index in matrix.positive.indices[start:stop]:
            item_users.setdefault(matrix.item_ids[item_index], set()).add(user_id)
    for user_id in matrix.user_ids:
        history = [matrix.item_ids[i] for i in matrix.positive_indices(user_id)]
        if not history:
            continue
        scores = ranker._user_scores(user_id, matrix.positive_indices(user_id))
        for item_id in matrix.item_ids:
            similarities = sorted(
                (cosine_similarity(item_users.get(item_id, set()), item_users[h]) for h in history),
                reverse=True,
            )
            expected = math.fsum(value for value in similarities[:3] if value > 0.0)
            assert scores[matrix.item_index[item_id]] == expected


def test_personalized_pagerank_matches_node_by_node_power_iteration() -> None:
    rows = _rows(count=300)
    ranker = PersonalizedPageRankRanker(damping=0.8, max_iter=25).fit(rows, 90)
    matrix = ranker.matrix
    positive = matrix.positive.tocoo()
    neighbors: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for user, item in zip(positive.row, positive.col, strict=True):
        neighbors.setdefault(("u", int(user)), []).append(("i", int(item)))
        neighbors.setdefault(("i", int(item)), []).append(("u", int(user)))
    for user_id in matrix.user_ids[:20]:
        restart = ("u", matrix.user_index[user_id])
        if restart not in neighbors:
            continue
        mass = {restart: 1.0}
        for _ in range(25):
            updated = {restart: 0.2}
            for node, value in mass.items():
                share = 0.8 * value / len(neighbors[node])
                for neighbor in neighbors[node]:
                    updated[neighbor] = updated.get(neighbor, 0.0) + share
            mass = updated
        scores = ranker._user_scores(user_id, matrix.positive_indices(user_id))
        expected = np.array([mass.get(("i", i), 0.0) for i in range(matrix.n_items)])
        np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-15)


def test_bpr_negative_samples_stay_outside_user_history() -> None:
    matrix = build_interaction_matrix(_rows(), 90)
    history = matrix.history
    counts = np.diff(history.indptr)
    keys = np.repeat(np.arange(matrix.n_users), counts) * matrix.n_items + history.indices
    users = np.repeat(np.flatnonzero(counts < matrix.n_items), 20)
    negatives = _sample_negatives(
        np.random.default_rng(1), users, history, keys.astype(np.int64), matrix.n_items
    )
    sampled = set((users * matrix.n_items + negatives).tolist())

    assert not sampled & set(keys.tolist())


def test_baselines_fit_and_rank_a_large_snapshot_quickly() -> None:
    generator = np.random.default_rng(0)
    count = 60_000
    table = pa.table(
        {
            "user_id": [f"u{value:06d}" for value in generator.integers(0, 30_000, count)],
            "item_id": [
                f"i{value:05d}" for value in (generator.pareto(1.0, count) * 40).astype(int) % 8000
            ],
            "rating": generator.choice([1.0, 3.0, 4.0, 5.0], count),
            "timestamp": generator.integers(0, 1000, count),
        }
    )
    began = time.perf_counter()
    matrix = build_interaction_matrix(table, 900)
    users = list(matrix.user_ids[:200])
    for ranker in (
        PopularityRanker(),
        ItemKNNRanker(),
        BPRMFRanker(epochs=2, embedding_dimensions=16),
        PersonalizedPageRankRanker(max_iter=20),
    ):
        ranker.fit(None, 900, matrix=matrix)
        if hasattr(ranker, "prepare"):
            ranker.prepare(users)
        for user_id in users:
            ranker.recommend(user_id, 20)

    assert time.perf_counter() - began < 60.0


@pytest.mark.parametrize("factory", RANKER_FACTORIES)
def test_saved_models_reload_with_identical_rankings(factory, tmp_path) -> None:
    from trustrec.recommenders.registry import load_baseline

    rows = _rows()
    ranker = factory().fit(rows, 90, snapshot_id="s")
    path = tmp_path / "model.npz"
    metadata = ranker.save(path)
    reloaded = load_baseline(path, build_interaction_matrix(list(reversed(rows)), 90))

    assert reloaded.model_hash == ranker.model_hash == metadata["model_hash"]
    for user_id in sorted({row["user_id"] for row in rows})[:15]:
        assert reloaded.recommend(user_id, 10) == ranker.recommend(user_id, 10)


def test_loading_a_model_on_other_data_is_rejected(tmp_path) -> None:
    from trustrec.recommenders.registry import load_baseline

    ranker = BPRMFRanker(epochs=2, embedding_dimensions=4).fit(_rows(), 90)
    path = tmp_path / "bpr.npz"
    ranker.save(path)

    with pytest.raises(ValueError, match="different data"):
        load_baseline(path, build_interaction_matrix(_rows(seed=12), 90))
