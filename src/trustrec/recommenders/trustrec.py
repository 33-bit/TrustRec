"""Adaptive TrustRec gate (T0)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .hybrid import HybridScore, adaptive_trustrec_details, adaptive_trustrec_scores, rank_scores


def rank_trustrec(
    mf_scores: Mapping[str, float],
    graph_scores: Mapping[str, float],
    aspect_scores: Mapping[str, float],
    support: Mapping[str, float],
    history_count: int,
    *,
    kappa: float = 5.0,
    g_max: float = 0.5,
    rho: float = 0.5,
    k: int | None = None,
) -> tuple[HybridScore, ...]:
    """Rank a shared candidate set with the adaptive TrustRec gate."""

    details = adaptive_trustrec_details(
        mf_scores,
        graph_scores,
        aspect_scores,
        support,
        history_count,
        kappa=kappa,
        g_max=g_max,
        rho=rho,
    )
    return tuple(
        details[item_id]
        for item_id, _ in rank_scores(
            {key: value.total_score for key, value in details.items()}, k=k
        )
    )


@dataclass(frozen=True)
class TrustRecRanker:
    """Reusable T0 scorer with validation-time gate parameters."""

    kappa: float = 5.0
    g_max: float = 0.5
    rho: float = 0.5

    def score(
        self,
        mf_scores: Mapping[str, float],
        graph_scores: Mapping[str, float],
        aspect_scores: Mapping[str, float],
        support: Mapping[str, float],
        history_count: int,
    ) -> dict[str, float]:
        return adaptive_trustrec_scores(
            mf_scores,
            graph_scores,
            aspect_scores,
            support,
            history_count,
            kappa=self.kappa,
            g_max=self.g_max,
            rho=self.rho,
        )

    def rank(
        self,
        mf_scores: Mapping[str, float],
        graph_scores: Mapping[str, float],
        aspect_scores: Mapping[str, float],
        support: Mapping[str, float],
        history_count: int,
        *,
        k: int | None = None,
    ) -> tuple[HybridScore, ...]:
        return rank_trustrec(
            mf_scores,
            graph_scores,
            aspect_scores,
            support,
            history_count,
            kappa=self.kappa,
            g_max=self.g_max,
            rho=self.rho,
            k=k,
        )


TrustRec = TrustRecRanker
