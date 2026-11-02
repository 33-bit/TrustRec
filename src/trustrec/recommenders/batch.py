"""Score many users with one fitted ranker, optionally in forked workers."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Sequence
from typing import Any

from .base import MatrixRanker

CANDIDATE_RULE = "catalog_minus_history"

# A forked worker reads the fitted ranker from this module global.
_WORKER_STATE: dict[str, Any] = {}


def _score_chunked(ranker: MatrixRanker, user_ids: Sequence[str], k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prepare = getattr(ranker, "prepare", None)
    clear = getattr(ranker, "clear_cache", None)
    step = getattr(ranker, "batch_size", 256) if prepare is not None else 256
    for start in range(0, len(user_ids), step):
        chunk = list(user_ids[start : start + step])
        began = time.perf_counter()
        if prepare is not None:
            prepare(chunk)
        shared_ms = (time.perf_counter() - began) * 1000.0 / max(len(chunk), 1)
        for user_id in chunk:
            began = time.perf_counter()
            recommendation = ranker.recommend(user_id, k)
            elapsed_ms = (time.perf_counter() - began) * 1000.0 + shared_ms
            rows.append(
                {
                    "user_id": user_id,
                    "ranked_item_ids": list(recommendation.item_ids),
                    "scores": list(recommendation.scores),
                    "candidate_rule": CANDIDATE_RULE,
                    "candidate_count": recommendation.candidate_count,
                    "candidate_signature": recommendation.candidate_signature,
                    "fallback_reason": recommendation.fallback_reason,
                    "latency_ms": elapsed_ms,
                }
            )
        if clear is not None:
            clear()
    return rows


def _worker_score(user_ids: list[str]) -> list[dict[str, Any]]:
    return _score_chunked(_WORKER_STATE["ranker"], user_ids, _WORKER_STATE["k"])


def score_users(
    ranker: MatrixRanker, user_ids: Sequence[str], k: int, *, workers: int = 1
) -> list[dict[str, Any]]:
    """Return one top-K row per user in input order. Workers need the fork start method."""

    if workers <= 1 or len(user_ids) < 2:
        return _score_chunked(ranker, user_ids, k)
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("workers > 1 requires the fork start method")
    chunk_size = max(1, -(-len(user_ids) // (workers * 8)))
    chunks = [list(user_ids[i : i + chunk_size]) for i in range(0, len(user_ids), chunk_size)]
    _WORKER_STATE.update(ranker=ranker, k=k)
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_worker_score, chunks, chunksize=1)
    finally:
        _WORKER_STATE.clear()
    return [row for part in parts for row in part]
