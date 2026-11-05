import random

import numpy as np
import pytest

from trustrec.graph import PersonalizedPageRankRanker
from trustrec.recommenders import BPRMFRanker, adaptive_trustrec_scores, fixed_hybrid_scores
from trustrec.recommenders.hybrid import percentile_normalize
from trustrec.recommenders.hybrid_batch import (
    HybridBatchScorer,
    HybridConfig,
    build_aspect_model,
    percentile_vector,
)
from trustrec.recommenders.matrix import build_interaction_matrix

ASPECTS = ("story", "graphics", "value")


def _data(seed: int = 5):
    generator = random.Random(seed)
    rows, evidence = [], []
    for position in range(700):
        user = f"u{generator.randint(0, 59):02d}"
        item = f"i{int(generator.paretovariate(1.3)) % 30:02d}"
        timestamp = generator.randint(0, 99)
        rows.append(
            {
                "review_id": f"r{position}",
                "user_id": user,
                "item_id": item,
                "rating": float(generator.randint(1, 5)),
                "timestamp": timestamp,
            }
        )
        if generator.random() < 0.6:
            evidence.append(
                {
                    "review_id": f"r{position}",
                    "item_id": item,
                    "author_id": user,
                    "timestamp": timestamp,
                    "aspect": generator.choice(ASPECTS),
                    "sentiment": generator.choice(["positive", "negative", "neutral"]),
                    "confidence": 1.0,
                }
            )
    return rows, evidence


def test_percentile_vector_matches_the_dictionary_normalizer() -> None:
    values = np.array([3.0, 1.0, 3.0, 2.0, 5.0])
    expected = percentile_normalize({f"i{i}": v for i, v in enumerate(values)})

    assert percentile_vector(values).tolist() == [expected[f"i{i}"] for i in range(5)]
    assert percentile_vector(np.array([2.0, 2.0])).tolist() == [0.5, 0.5]


def test_aspect_model_follows_the_design_formulas() -> None:
    rows, evidence = _data()
    matrix = build_interaction_matrix(rows, 100)
    model = build_aspect_model(evidence, matrix, 100, aspects=ASPECTS, alpha=2.0)

    assert model.prior_distribution.sum() == pytest.approx(1.0)
    user = matrix.user_index["u01"]
    weights = model.user_weights(user)
    counts = model.user_counts.get(user, np.zeros(3))
    expected = (counts + 2.0 * model.prior_distribution) / (counts.sum() + 2.0)
    np.testing.assert_allclose(weights, expected)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(model.item_support >= 0.0) and np.all(model.item_support < 1.0)


@pytest.mark.parametrize(
    "config",
    [
        HybridConfig("h0_fixed_hybrid", mf_weight=0.4, graph_weight=0.3, aspect_weight=0.3),
        HybridConfig("t0_trustrec", kappa=3.0, g_max=0.6, rho=0.4),
    ],
)
def test_batch_hybrid_matches_the_dictionary_formulas(config: HybridConfig) -> None:
    rows, evidence = _data()
    matrix = build_interaction_matrix(rows, 100)
    mf = BPRMFRanker(epochs=3, embedding_dimensions=4).fit(None, 100, matrix=matrix)
    graph = PersonalizedPageRankRanker(max_iter=20).fit(None, 100, matrix=matrix)
    aspect_model = build_aspect_model(evidence, matrix, 100, aspects=ASPECTS)
    scorer = HybridBatchScorer(matrix, mf, graph, aspect_model)
    for user_id in matrix.user_ids[:20]:
        parts = scorer.components(user_id)
        ids = [matrix.item_ids[i] for i in parts.candidates]
        raw_mf, _ = scorer._component(
            mf, user_id, matrix.positive_indices(user_id), parts.candidates
        )
        raw_graph, _ = scorer._component(
            graph, user_id, matrix.positive_indices(user_id), parts.candidates
        )
        aspect, support = aspect_model.scores(matrix.user_index[user_id])
        mf_map = dict(zip(ids, raw_mf, strict=True))
        graph_map = dict(zip(ids, raw_graph, strict=True))
        aspect_map = dict(zip(ids, aspect[parts.candidates], strict=True))
        if config.model_id == "h0_fixed_hybrid":
            expected = fixed_hybrid_scores(mf_map, graph_map, aspect_map, **config.parameters())
        else:
            support_map = dict(zip(ids, support[parts.candidates], strict=True))
            expected = adaptive_trustrec_scores(
                mf_map,
                graph_map,
                aspect_map,
                support_map,
                parts.history_count,
                **config.parameters(),
            )
        total, _ = scorer.totals(parts, config)
        np.testing.assert_allclose(total, [expected[item] for item in ids], rtol=1e-12)
        ((top, _),) = scorer.recommend_many_configs(user_id, [config], 5)
        ranked = sorted(expected, key=lambda item: (-expected[item], item))[:5]
        assert list(top.item_ids) == ranked


def test_hybrid_config_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        HybridConfig("h0_fixed_hybrid", mf_weight=0.5, graph_weight=0.5, aspect_weight=0.5)
    with pytest.raises(ValueError, match="gate"):
        HybridConfig("t0_trustrec", g_max=1.5)


def _scorer(**settings):
    rows, evidence = _data()
    rows.append(
        {"review_id": "cold", "user_id": "ucold", "item_id": "i00", "rating": 1.0, "timestamp": 5}
    )
    matrix = build_interaction_matrix(rows, 100)
    mf = BPRMFRanker(embedding_dimensions=4, epochs=2, seed=3)
    mf.fit(None, 100, "snap", matrix=matrix)
    graph = PersonalizedPageRankRanker(max_iter=20)
    graph.fit(None, 100, "snap", matrix=matrix)
    aspects = build_aspect_model(evidence, matrix, 100, aspects=ASPECTS)
    return matrix, HybridBatchScorer(matrix, mf, graph, aspects, **settings)


def test_minmax_vector_keeps_gaps_and_handles_constants() -> None:
    from trustrec.recommenders.hybrid_batch import minmax_vector

    assert minmax_vector(np.array([0.0, 1.0, 1000.0])).tolist() == [0.0, 0.001, 1.0]
    assert minmax_vector(np.array([4.0, 4.0])).tolist() == [0.5, 0.5]
    assert minmax_vector(np.array([])).size == 0


def test_cold_fallback_returns_the_baseline_order_without_aspects() -> None:
    from trustrec.recommenders.popularity import PopularityRanker

    configs = [
        HybridConfig("h0_fixed_hybrid", mf_weight=0.2, graph_weight=0.4, aspect_weight=0.4),
        HybridConfig("t0_trustrec", g_max=0.75),
    ]
    matrix, blend = _scorer()
    _, fallback = _scorer(cold_start="fallback", normalization="minmax")
    popularity = PopularityRanker().fit(None, 100, "snap", matrix=matrix)
    expected = popularity.recommend("ucold", k=10).item_ids

    results = fallback.recommend_many_configs("ucold", configs, 10)

    assert all(rec.item_ids == expected for rec, _ in results)
    assert all(
        gate == 0.0 and rec.fallback_reason == "no_positive_history" for rec, gate in results
    )
    assert blend.recommend_many_configs("ucold", configs[:1], 10)[0][0].item_ids != expected
    warm = next(user for user in matrix.user_ids if matrix.positive_indices(user).size)
    assert fallback.recommend_many_configs(warm, configs, 10)[0][0].fallback_reason is None
    with pytest.raises(ValueError, match="normalization"):
        _scorer(normalization="zscore")


def test_recency_term_enters_h0_and_the_t0_base() -> None:
    matrix, plain = _scorer(normalization="minmax")
    recency = matrix.recent_popularity(1).astype(float)
    _, scorer = _scorer(normalization="minmax", recency_scores=recency)
    warm = next(user for user in matrix.user_ids if matrix.positive_indices(user).size)
    parts = scorer.components(warm)
    h0 = HybridConfig(
        "h0_fixed_hybrid", mf_weight=0.1, graph_weight=0.2, recency_weight=0.5, aspect_weight=0.2
    )
    t0 = HybridConfig("t0_trustrec", g_max=0.0, rho=0.5, tau=1.0)

    h0_total, _ = scorer.totals(parts, h0)
    t0_total, gate = scorer.totals(parts, t0)

    expected = 0.1 * parts.mf + 0.2 * parts.graph + 0.5 * parts.recency + 0.2 * parts.aspect
    np.testing.assert_allclose(h0_total, expected)
    np.testing.assert_allclose(t0_total, parts.recency)
    assert not gate.any()
    np.testing.assert_allclose(
        parts.recency,
        minmax_scaled := (recency[parts.candidates] - recency[parts.candidates].min())
        / (recency[parts.candidates].max() - recency[parts.candidates].min()),
    )
    assert minmax_scaled.max() == 1.0
    with pytest.raises(ValueError, match="recency_scores"):
        plain.recommend_many_configs(warm, [h0], 5)
    with pytest.raises(ValueError, match="sum to 1"):
        HybridConfig("h0_fixed_hybrid", mf_weight=0.5, graph_weight=0.5, recency_weight=0.5)
