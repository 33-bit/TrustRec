import pytest

from trustrec.recommenders.hybrid import (
    adaptive_trustrec_scores,
    fixed_hybrid_scores,
    percentile_normalize,
)


def test_percentile_normalization_uses_average_ties_and_shared_bounds() -> None:
    normalized = percentile_normalize({"a": 1.0, "b": 2.0, "c": 2.0, "d": 4.0})
    assert normalized == {
        "a": pytest.approx(0.0),
        "b": pytest.approx(0.5),
        "c": pytest.approx(0.5),
        "d": pytest.approx(1.0),
    }
    assert percentile_normalize({"a": 4.0, "b": 4.0}) == {"a": 0.5, "b": 0.5}


def test_fixed_and_adaptive_hybrids_use_the_same_normalized_components() -> None:
    mf = {"a": 0.1, "b": 0.8}
    graph = {"a": 0.9, "b": 0.2}
    aspect = {"a": 0.2, "b": 0.7}
    fixed = fixed_hybrid_scores(
        mf, graph, aspect, mf_weight=0.5, graph_weight=0.25, aspect_weight=0.25
    )
    adaptive = adaptive_trustrec_scores(
        mf,
        graph,
        aspect,
        support={"a": 1.0, "b": 0.0},
        history_count=0,
        kappa=5,
        g_max=0.5,
        rho=0.5,
    )
    assert fixed["a"] == pytest.approx(0.25)
    assert fixed["b"] == pytest.approx(0.75)
    # Candidate a has g=.5 and candidate b has g=0.
    assert adaptive["a"] == pytest.approx(0.25)
    assert adaptive["b"] == pytest.approx(0.5)


def test_hybrid_rejects_mismatched_candidates_and_invalid_gate() -> None:
    with pytest.raises(ValueError, match="same candidate IDs"):
        fixed_hybrid_scores({"a": 1}, {"a": 1}, {"b": 1})
    with pytest.raises(ValueError, match="g_max"):
        adaptive_trustrec_scores({"a": 1}, {"a": 1}, {"a": 1}, {"a": 1}, 0, g_max=1.1)
