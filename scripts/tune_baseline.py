"""Tune one ranking baseline on the validation cutoff with a fixed, recorded trial budget."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trustrec.evaluation.protocol import build_evaluation_cases, compare_model_rankings
from trustrec.recommenders.batch import score_users
from trustrec.recommenders.contracts import stable_model_hash
from trustrec.recommenders.matrix import build_interaction_matrix
from trustrec.recommenders.popularity import PopularityRanker
from trustrec.recommenders.registry import ranker_class

try:
    from scripts.run_rankings_batch import (
        _assert_completed_manifest_is_absent,
        _load_manifest,
        _read_table,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
    )
except ModuleNotFoundError:  # run as ``python scripts/tune_baseline.py``
    from run_rankings_batch import (  # type: ignore[no-redef]
        _assert_completed_manifest_is_absent,
        _load_manifest,
        _read_table,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
    )

INTERACTIONS_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}
TARGETS_BY_CUTOFF = {"t0": "validation_targets", "t1": "test_targets"}
SELECTION_METRIC = "ndcg@10"
K_VALUES = (5, 10, 20)


def _evaluation_rows(
    rows: Sequence[dict[str, Any]], snapshot_id: str, cutoff_timestamp: Any
) -> list[dict[str, Any]]:
    return [
        {**row, "snapshot_id": snapshot_id, "cutoff_timestamp": cutoff_timestamp} for row in rows
    ]


def _summary(evaluation: Any, metric: str) -> dict[str, Any]:
    bootstrap = evaluation.bootstrap.get(metric)
    return {
        "estimate": evaluation.metrics.get(metric),
        "interval": None if bootstrap is None else [bootstrap.lower, bootstrap.upper],
        "delta_vs_popularity": None if bootstrap is None else bootstrap.delta_estimate,
        "delta_interval": None
        if bootstrap is None or bootstrap.delta_lower is None
        else [bootstrap.delta_lower, bootstrap.delta_upper],
    }


DEFAULT_GRIDS: dict[str, list[dict[str, Any]]] = {
    "b1_item_knn": [{"n_neighbors": value} for value in (None, 5, 10, 20, 50, 100, 200, 500, 1000)],
    "b2_bpr_mf": [
        {
            "embedding_dimensions": 64,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "epochs": 30,
            "negative_samples": 1,
            "batch_size": 1024,
        }
        for learning_rate in (0.01, 0.05, 0.1)
        for regularization in (0.0001, 0.001, 0.01)
    ],
    "b3_ppr": [
        {"damping": damping, "max_iter": max_iter, "tol": 1e-12, "batch_size": 64}
        for damping in (0.5, 0.7, 0.85)
        for max_iter in (10, 30, 100)
    ],
    "b5_recent_popularity": [
        {"window_days": days} for days in (30, 60, 90, 180, 365, 540, 730, 1095, 1825)
    ],
}
TUNABLE_MODELS = tuple(DEFAULT_GRIDS)


def tune_baseline(
    manifest_path: Path,
    *,
    model_id: str,
    output_path: Path,
    grid: Sequence[Mapping[str, Any]] | None = None,
    cutoff_name: str = "t0",
    positive_threshold: float = 4.0,
    seed: int = 7,
    workers: int = 1,
    bootstrap_samples: int = 200,
    save_best_path: Path | None = None,
) -> dict[str, Any]:
    """Fit every grid point once, score validation users, and keep the best by NDCG@10."""

    if cutoff_name != "t0":
        raise ValueError("tune on the validation cutoff t0 only; t1 is the test cutoff")
    if model_id not in TUNABLE_MODELS:
        raise ValueError(f"unsupported model_id: {model_id}")
    grid = [dict(point) for point in (grid if grid is not None else DEFAULT_GRIDS[model_id])]
    if not grid:
        raise ValueError("grid must contain at least one trial")
    _assert_completed_manifest_is_absent(output_path)
    manifest = _load_manifest(manifest_path)
    snapshot_id = str(manifest["snapshot_id"])
    cutoff_timestamp = manifest["cutoffs"][cutoff_name]
    interaction_name = INTERACTIONS_BY_CUTOFF[cutoff_name]
    target_name = TARGETS_BY_CUTOFF[cutoff_name]
    interaction_path = _resolve_artifact(manifest_path, manifest, interaction_name)
    target_path = _resolve_artifact(manifest_path, manifest, target_name)
    source_hashes = {
        interaction_name: _verified_hash(manifest, interaction_name, interaction_path),
        target_name: _verified_hash(manifest, target_name, target_path),
    }

    began = time.perf_counter()
    interactions = _read_table(
        interaction_path, ("review_id", "user_id", "item_id", "rating", "timestamp")
    )
    matrix = build_interaction_matrix(interactions, cutoff_timestamp, positive_threshold)
    dataset = build_evaluation_cases(
        interactions.to_pylist(),
        pq.read_table(target_path).to_pylist(),
        snapshot_id=snapshot_id,
        cutoff_timestamp=cutoff_timestamp,
        positive_threshold=positive_threshold,
    )
    del interactions
    users = [case.user_id for case in dataset.cases]
    setup_ms = (time.perf_counter() - began) * 1000.0
    k = max(K_VALUES)

    popularity = PopularityRanker(positive_threshold=positive_threshold)
    popularity.fit(None, cutoff_timestamp, snapshot_id, matrix=matrix)
    popularity_rows = score_users(popularity, users, k, workers=workers)
    popularity_top10 = {row["user_id"]: row["ranked_item_ids"][:10] for row in popularity_rows}
    reference = _evaluation_rows(popularity_rows, snapshot_id, cutoff_timestamp)

    trials: list[dict[str, Any]] = []
    best: tuple[float, Any, dict[str, Any]] | None = None
    for grid_point in grid:
        parameters = {**grid_point, "positive_threshold": positive_threshold, "seed": seed}
        ranker = ranker_class(model_id)(**parameters)
        started = time.perf_counter()
        ranker.fit(None, cutoff_timestamp, snapshot_id, matrix=matrix)
        fit_ms = (time.perf_counter() - started) * 1000.0
        rows = score_users(ranker, users, k, workers=workers)
        personalized = [row for row in rows if row["fallback_reason"] is None]
        same_as_popularity = sum(
            row["ranked_item_ids"][:10] == popularity_top10[row["user_id"]] for row in personalized
        )
        evaluations = compare_model_rankings(
            dataset,
            {
                "b0_most_popular": reference,
                model_id: _evaluation_rows(rows, snapshot_id, cutoff_timestamp),
            },
            k_values=K_VALUES,
            bootstrap_seed=seed,
            bootstrap_samples=bootstrap_samples,
            reference_model="b0_most_popular",
        )
        evaluation = evaluations[model_id]
        trial = {
            "parameters": parameters,
            "model_hash": ranker.model_hash,
            "fit_ms": fit_ms,
            "metrics": {
                metric: _summary(evaluation, metric)
                for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
            },
            "fallback_counts": dict(Counter(row["fallback_reason"] or "none" for row in rows)),
            "personalized_users": len(personalized),
            "top10_equal_to_popularity": same_as_popularity,
        }
        trials.append(trial)
        print(
            json.dumps(
                {
                    "trial": len(trials),
                    "of": len(grid),
                    **grid_point,
                    SELECTION_METRIC: trial["metrics"][SELECTION_METRIC]["estimate"],
                    "top10_equal_to_popularity": same_as_popularity,
                }
            ),
            flush=True,
        )
        score = evaluation.metrics[SELECTION_METRIC]
        if best is None or score > best[0]:
            best = (score, ranker, trial)
        else:
            del ranker

    assert best is not None
    best_trial = best[2]
    model_file = None
    if save_best_path is not None:
        best[1].save(save_best_path)
        model_file = {"path": str(save_best_path), "sha256": _sha256_file(save_best_path)}
    popularity_eval = compare_model_rankings(
        dataset, {"b0_most_popular": reference}, k_values=K_VALUES, bootstrap_samples=20
    )["b0_most_popular"]
    result = {
        "schema_version": 1,
        "task": "baseline_tuning",
        "model_id": model_id,
        "status": "complete",
        "run_id": stable_model_hash(
            {
                "snapshot_id": snapshot_id,
                "model_id": model_id,
                "grid": grid,
                "seed": seed,
                "cutoff": cutoff_name,
            }
        )[:16],
        "snapshot_id": snapshot_id,
        "dataset_hash": manifest["dataset_hash"],
        "data_fingerprint": matrix.fingerprint,
        "cutoff_name": cutoff_name,
        "cutoff_timestamp": cutoff_timestamp,
        "candidate_set_hash": dataset.candidate_set_hash,
        "eligible_users": len(users),
        "source_artifact_sha256": source_hashes,
        "selection_metric": SELECTION_METRIC,
        "tuning_budget": {
            "trials": len(grid),
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
        },
        "grid": grid,
        "popularity_reference": {
            metric: popularity_eval.metrics.get(metric) for metric in ("ndcg@10", "recall@10")
        },
        "best": best_trial,
        "best_model_file": model_file,
        "trials": trials,
        "timing_ms": {"setup": setup_ms},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=TUNABLE_MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grid", type=Path, help="JSON list of parameter objects; defaults to DEFAULT_GRIDS"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--save-best", type=Path)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text(encoding="utf-8")) if args.grid else None
    result = tune_baseline(
        args.snapshot_manifest,
        model_id=args.model,
        output_path=args.output,
        grid=grid,
        seed=args.seed,
        workers=args.workers,
        bootstrap_samples=args.bootstrap_samples,
        save_best_path=args.save_best,
    )
    print(json.dumps({"best": result["best"]["parameters"], "trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
