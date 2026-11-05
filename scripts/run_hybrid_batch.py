"""Tune or run H0 and T0 for every evaluation user from fitted components and evidence."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trustrec.evaluation.protocol import build_evaluation_cases, compare_model_rankings
from trustrec.graph.personalized_pagerank import PersonalizedPageRankRanker
from trustrec.recommenders.batch import score_users
from trustrec.recommenders.bpr_mf import BPRMFRanker
from trustrec.recommenders.contracts import stable_model_hash
from trustrec.recommenders.hybrid_batch import (
    COLD_START_POLICIES,
    NORMALIZATIONS,
    HybridBatchScorer,
    HybridConfig,
    build_aspect_model,
    score_hybrid_configs,
)
from trustrec.recommenders.matrix import build_interaction_matrix
from trustrec.recommenders.popularity import PopularityRanker
from trustrec.recommenders.registry import load_baseline

try:
    from scripts.run_rankings_batch import (
        _assert_completed_manifest_is_absent,
        _load_manifest,
        _read_table,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
        evaluation_user_ids,
    )
except ModuleNotFoundError:  # run as ``python scripts/run_hybrid_batch.py``
    from run_rankings_batch import (  # type: ignore[no-redef]
        _assert_completed_manifest_is_absent,
        _load_manifest,
        _read_table,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
        evaluation_user_ids,
    )

INTERACTIONS_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}
TARGETS_BY_CUTOFF = {"t0": "validation_targets", "t1": "test_targets"}
K_VALUES = (5, 10, 20)
DEFAULT_SCORER_SETTINGS: dict[str, Any] = {
    "normalization": "percentile",
    "cold_start": "blend",
    "cold_window_days": None,
}
DEFAULT_GRID = [
    *(
        HybridConfig("h0_fixed_hybrid", mf_weight=m, graph_weight=g, aspect_weight=a)
        for m, g, a in (
            (0.5, 0.25, 0.25),
            (0.4, 0.4, 0.2),
            (0.25, 0.5, 0.25),
            (0.2, 0.6, 0.2),
            (0.4, 0.3, 0.3),
            (0.3, 0.3, 0.4),
            (0.2, 0.4, 0.4),
            (0.1, 0.6, 0.3),
            (0.0, 0.7, 0.3),
        )
    ),
    *(
        HybridConfig("t0_trustrec", kappa=5.0, g_max=g_max, rho=rho)
        for g_max in (0.25, 0.5, 0.75)
        for rho in (0.2, 0.5, 0.8)
    ),
]


def _read_evidence(path: Path) -> Iterator[dict[str, Any]]:
    fields = ("review_id", "item_id", "author_id", "timestamp", "aspect", "sentiment")
    fields += ("sentiment_score", "confidence", "duplicate_group")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield {key: row[key] for key in fields if key in row}


def _components(
    manifest_path: Path,
    cutoff_name: str,
    evidence_path: Path,
    *,
    mf_parameters: dict[str, Any],
    graph_parameters: dict[str, Any],
    mf_model_path: Path | None,
    positive_threshold: float,
    scorer_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = {**DEFAULT_SCORER_SETTINGS, **(scorer_settings or {})}
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
        "evidence": _sha256_file(evidence_path),
    }
    interactions = _read_table(
        interaction_path, ("review_id", "user_id", "item_id", "rating", "timestamp")
    )
    matrix = build_interaction_matrix(interactions, cutoff_timestamp, positive_threshold)
    began = time.perf_counter()
    if mf_model_path is not None:
        mf = load_baseline(mf_model_path, matrix)
        if not isinstance(mf, BPRMFRanker):
            raise ValueError("--mf-model must hold a BPR MF model")
    else:
        mf = BPRMFRanker(positive_threshold=positive_threshold, **mf_parameters)
        mf.fit(None, cutoff_timestamp, snapshot_id, matrix=matrix)
    graph = PersonalizedPageRankRanker(positive_threshold=positive_threshold, **graph_parameters)
    graph.fit(None, cutoff_timestamp, snapshot_id, matrix=matrix)
    aspect_model = build_aspect_model(_read_evidence(evidence_path), matrix, cutoff_timestamp)
    window = settings["cold_window_days"]
    cold_scores = None if window is None else matrix.recent_popularity(window).astype(float)
    scorer = HybridBatchScorer(
        matrix,
        mf,
        graph,
        aspect_model,
        normalization=settings["normalization"],
        cold_start=settings["cold_start"],
        cold_scores=cold_scores,
    )
    return {
        "manifest": manifest,
        "snapshot_id": snapshot_id,
        "cutoff_timestamp": cutoff_timestamp,
        "interactions": interactions,
        "target_path": target_path,
        "matrix": matrix,
        "scorer": scorer,
        "scorer_settings": settings,
        "source_hashes": source_hashes,
        "component_hashes": {"mf": mf.model_hash, "graph": graph.model_hash},
        "aspect_summary": {
            "evidence_rows": aspect_model.evidence_rows,
            "profiled_items": aspect_model.profiled_items,
            "users_with_aspect_history": len(aspect_model.user_counts),
            "prior_distribution": dict(
                zip(aspect_model.aspects, aspect_model.prior_distribution.tolist(), strict=True)
            ),
        },
        "fit_ms": (time.perf_counter() - began) * 1000.0,
    }


def _with_lineage(rows: Sequence[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "snapshot_id": context["snapshot_id"],
            "cutoff_timestamp": context["cutoff_timestamp"],
        }
        for row in rows
    ]


def tune_hybrids(
    manifest_path: Path,
    *,
    evidence_path: Path,
    output_path: Path,
    grid: Sequence[HybridConfig] = DEFAULT_GRID,
    mf_parameters: dict[str, Any] | None = None,
    graph_parameters: dict[str, Any] | None = None,
    mf_model_path: Path | None = None,
    positive_threshold: float = 4.0,
    workers: int = 1,
    bootstrap_samples: int = 200,
    seed: int = 7,
    scorer_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every grid point on ``t0`` and record validation metrics against popularity."""

    _assert_completed_manifest_is_absent(output_path)
    context = _components(
        manifest_path,
        "t0",
        evidence_path,
        mf_parameters=mf_parameters or {},
        graph_parameters=graph_parameters or {},
        mf_model_path=mf_model_path,
        positive_threshold=positive_threshold,
        scorer_settings=scorer_settings,
    )
    dataset = build_evaluation_cases(
        context.pop("interactions").to_pylist(),
        pq.read_table(context["target_path"]).to_pylist(),
        snapshot_id=context["snapshot_id"],
        cutoff_timestamp=context["cutoff_timestamp"],
        positive_threshold=positive_threshold,
    )
    users = [case.user_id for case in dataset.cases]
    popularity = PopularityRanker(positive_threshold=positive_threshold).fit(
        None, context["cutoff_timestamp"], context["snapshot_id"], matrix=context["matrix"]
    )
    reference = _with_lineage(score_users(popularity, users, max(K_VALUES)), context)
    began = time.perf_counter()
    all_rows = score_hybrid_configs(context["scorer"], users, grid, max(K_VALUES), workers=workers)
    scoring_ms = (time.perf_counter() - began) * 1000.0
    trials = []
    for config, rows in zip(grid, all_rows, strict=True):
        evaluation = compare_model_rankings(
            dataset,
            {"b0_most_popular": reference, config.model_id: _with_lineage(rows, context)},
            k_values=K_VALUES,
            bootstrap_seed=seed,
            bootstrap_samples=bootstrap_samples,
            reference_model="b0_most_popular",
        )[config.model_id]
        summary = evaluation.bootstrap["ndcg@10"]
        trials.append(
            {
                "model_id": config.model_id,
                "parameters": config.parameters(),
                "ndcg@10": evaluation.metrics["ndcg@10"],
                "recall@10": evaluation.metrics["recall@10"],
                "ndcg@10_interval": [summary.lower, summary.upper],
                "delta_vs_popularity": summary.delta_estimate,
                "delta_interval": [summary.delta_lower, summary.delta_upper],
                "mean_gate": sum(row["mean_gate"] for row in rows) / max(len(rows), 1),
            }
        )
        print(json.dumps(trials[-1]), flush=True)
    best = {}
    for model_id in sorted({config.model_id for config in grid}):
        candidates = [trial for trial in trials if trial["model_id"] == model_id]
        best[model_id] = max(candidates, key=lambda trial: trial["ndcg@10"])
    result = {
        "schema_version": 1,
        "task": "hybrid_tuning",
        "status": "complete",
        "snapshot_id": context["snapshot_id"],
        "dataset_hash": context["manifest"]["dataset_hash"],
        "cutoff_name": "t0",
        "candidate_set_hash": dataset.candidate_set_hash,
        "eligible_users": len(users),
        "source_artifact_sha256": context["source_hashes"],
        "component_model_hashes": context["component_hashes"],
        "mf_parameters": mf_parameters,
        "graph_parameters": graph_parameters,
        "scorer_settings": context["scorer_settings"],
        "aspect_summary": context["aspect_summary"],
        "selection_metric": "ndcg@10",
        "tuning_budget": {
            model_id: sum(1 for config in grid if config.model_id == model_id)
            for model_id in sorted({config.model_id for config in grid})
        },
        "best": best,
        "trials": trials,
        "timing_ms": {"components": context["fit_ms"], "scoring": scoring_ms},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_hybrids(
    manifest_path: Path,
    *,
    cutoff_name: str,
    evidence_path: Path,
    configs: Sequence[HybridConfig],
    output_paths: Sequence[Path],
    mf_parameters: dict[str, Any] | None = None,
    graph_parameters: dict[str, Any] | None = None,
    mf_model_path: Path | None = None,
    positive_threshold: float = 4.0,
    k: int = 100,
    workers: int = 1,
    user_ids: Sequence[str] | None = None,
    scorer_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Write one JSON Lines ranking file and manifest per configuration."""

    if len(configs) != len(output_paths):
        raise ValueError("configs and output_paths must have the same length")
    manifest_paths = [path.with_suffix(".manifest.json") for path in output_paths]
    for path in manifest_paths:
        _assert_completed_manifest_is_absent(path)
    context = _components(
        manifest_path,
        cutoff_name,
        evidence_path,
        mf_parameters=mf_parameters or {},
        graph_parameters=graph_parameters or {},
        mf_model_path=mf_model_path,
        positive_threshold=positive_threshold,
        scorer_settings=scorer_settings,
    )
    context.pop("interactions")
    if user_ids is None:
        user_ids = evaluation_user_ids(
            context["matrix"],
            _read_table(context["target_path"], ("user_id", "item_id", "rating", "timestamp")),
            positive_threshold,
        )
    began = time.perf_counter()
    all_rows = score_hybrid_configs(context["scorer"], list(user_ids), configs, k, workers=workers)
    scoring_ms = (time.perf_counter() - began) * 1000.0
    manifests = []
    for config, rows, output_path, run_manifest_path in zip(
        configs, all_rows, output_paths, manifest_paths, strict=True
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                record = {"model_id": config.model_id, **row}
                record = {**record, **_with_lineage([record], context)[0]}
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        configuration = {**asdict(config), "k": k, **context["scorer_settings"]}
        run_manifest = {
            "schema_version": 1,
            "task": "batch_rankings",
            "model_id": config.model_id,
            "snapshot_id": context["snapshot_id"],
            "dataset_hash": context["manifest"]["dataset_hash"],
            "data_fingerprint": context["matrix"].fingerprint,
            "cutoff_name": cutoff_name,
            "cutoff_timestamp": context["cutoff_timestamp"],
            "parameters": config.parameters(),
            "configuration_hash": stable_model_hash(configuration),
            "component_model_hashes": context["component_hashes"],
            "mf_parameters": mf_parameters,
            "graph_parameters": graph_parameters,
            "scorer_settings": context["scorer_settings"],
            "aspect_summary": context["aspect_summary"],
            "source_artifact_sha256": context["source_hashes"],
            "users": len(rows),
            "fallback_counts": dict(Counter(row["fallback_reason"] or "none" for row in rows)),
            "mean_gate": sum(row["mean_gate"] for row in rows) / max(len(rows), 1),
            "timing_ms": {"components": context["fit_ms"], "scoring": scoring_ms},
            "output_path": str(output_path),
            "output_sha256": _sha256_file(output_path),
            "status": "complete",
        }
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifests.append(run_manifest)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("tune", "run"):
        mode = sub.add_parser(name)
        mode.add_argument("--snapshot-manifest", type=Path, required=True)
        mode.add_argument("--evidence", type=Path, required=True)
        mode.add_argument("--mf-params", type=json.loads, default={}, help="BPR MF JSON parameters")
        mode.add_argument("--mf-model", type=Path, help="saved BPR MF model for this cutoff")
        mode.add_argument("--graph-params", type=json.loads, default={}, help="PPR JSON parameters")
        mode.add_argument("--workers", type=int, default=1)
        mode.add_argument("--normalization", choices=NORMALIZATIONS, default="percentile")
        mode.add_argument("--cold-start", choices=COLD_START_POLICIES, default="blend")
        mode.add_argument(
            "--cold-window-days",
            type=float,
            help="with --cold-start fallback, rank cold users by B5 with this window (else B0)",
        )
    tune = sub.choices["tune"]
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--bootstrap-samples", type=int, default=200)
    run = sub.choices["run"]
    run.add_argument("--cutoff", choices=tuple(INTERACTIONS_BY_CUTOFF), default="t1")
    run.add_argument("--h0", type=json.loads, help="H0 JSON weights")
    run.add_argument("--t0", type=json.loads, help="T0 JSON gate parameters")
    run.add_argument("--h0-output", type=Path)
    run.add_argument("--t0-output", type=Path)
    run.add_argument("--k", type=int, default=100)
    args = parser.parse_args()
    common = {
        "evidence_path": args.evidence,
        "mf_parameters": args.mf_params,
        "graph_parameters": args.graph_params,
        "mf_model_path": args.mf_model,
        "workers": args.workers,
        "scorer_settings": {
            "normalization": args.normalization,
            "cold_start": args.cold_start,
            "cold_window_days": args.cold_window_days,
        },
    }
    if args.mode == "tune":
        result = tune_hybrids(
            args.snapshot_manifest,
            output_path=args.output,
            bootstrap_samples=args.bootstrap_samples,
            **common,
        )
        print(json.dumps({"best": result["best"]}))
        return
    configs, outputs = [], []
    if args.h0 is not None:
        configs.append(HybridConfig("h0_fixed_hybrid", **args.h0))
        outputs.append(args.h0_output)
    if args.t0 is not None:
        configs.append(HybridConfig("t0_trustrec", **args.t0))
        outputs.append(args.t0_output)
    if not configs or any(path is None for path in outputs):
        raise SystemExit("pass --h0/--h0-output and/or --t0/--t0-output")
    manifests = run_hybrids(
        args.snapshot_manifest,
        cutoff_name=args.cutoff,
        configs=configs,
        output_paths=outputs,
        k=args.k,
        **common,
    )
    print(
        json.dumps([{key: m[key] for key in ("model_id", "users", "mean_gate")} for m in manifests])
    )


if __name__ == "__main__":
    main()
