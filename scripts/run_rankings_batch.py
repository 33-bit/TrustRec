"""Fit one ranking baseline once and write top-K rows for every evaluation user."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trustrec.recommenders.base import MatrixRanker
from trustrec.recommenders.contracts import stable_model_hash, timestamp_value
from trustrec.recommenders.matrix import InteractionMatrix, build_interaction_matrix
from trustrec.recommenders.registry import (
    BASELINE_MODEL_IDS,
    BaselineParameters,
    create_baseline,
    effective_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}
TARGETS_BY_CUTOFF = {"t0": "validation_targets", "t1": "test_targets"}
CANDIDATE_RULE = "catalog_minus_history"
INTERACTION_COLUMNS = ("review_id", "user_id", "item_id", "rating", "timestamp")

# A forked worker reads the fitted ranker from this module global.
_WORKER_STATE: dict[str, Any] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"snapshot manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for field in ("snapshot_id", "dataset_hash", "cutoffs", "artifact_paths"):
        if not manifest.get(field):
            raise ValueError(f"snapshot manifest is missing {field}")
    return manifest


def _resolve_artifact(manifest_path: Path, manifest: dict[str, Any], name: str) -> Path:
    value = manifest["artifact_paths"].get(name)
    if not value:
        raise ValueError(f"snapshot manifest is missing artifact_paths.{name}")
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (
        manifest_path.resolve().parent / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} artifact does not exist: {value}")


def _verified_hash(manifest: dict[str, Any], name: str, path: Path) -> str:
    actual = _sha256_file(path)
    expected = manifest.get("artifact_sha256", {}).get(name)
    if expected and actual != expected:
        raise ValueError(f"{name} hash does not match the snapshot manifest")
    return actual


def _read_table(path: Path, columns: Sequence[str]) -> pa.Table:
    available = set(pq.read_schema(path).names)
    return pq.read_table(path, columns=[name for name in columns if name in available])


def evaluation_user_ids(
    matrix: InteractionMatrix, targets: pa.Table, positive_threshold: float
) -> list[str]:
    """Return users with at least one eligible target, using the T5.1 target rules.

    A target is eligible when it is at or after the cutoff, has a rating at or
    above the threshold, names a catalog item, and is outside the user's
    pre-cutoff history. Users also need a non-empty candidate pool.
    """

    frame = targets.select(
        [
            name
            for name in ("user_id", "item_id", "rating", "timestamp")
            if name in targets.schema.names
        ]
    ).to_pandas()
    for column in ("user_id", "item_id", "rating", "timestamp"):
        if column not in frame.columns:
            raise ValueError(f"target rows require {column}")
    cutoff = timestamp_value(matrix.cutoff_timestamp)
    if frame["timestamp"].dtype.kind in "iuf":
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    else:
        timestamps = np.array([timestamp_value(value) for value in frame["timestamp"]])
    ratings = frame["rating"].to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        keep = (timestamps >= cutoff) & (ratings >= positive_threshold)
    eligible: set[str] = set()
    for user_id, item_id in zip(
        frame["user_id"].to_numpy()[keep], frame["item_id"].to_numpy()[keep], strict=True
    ):
        if user_id in eligible:
            continue
        item_index = matrix.item_index.get(item_id)
        if item_index is None:
            continue
        history = matrix.history_indices(user_id)
        if history.size >= matrix.n_items:
            continue
        position = np.searchsorted(history, item_index)
        if position < history.size and history[position] == item_index:
            continue
        eligible.add(user_id)
    return sorted(eligible)


def _score_users(ranker: MatrixRanker, user_ids: Sequence[str], k: int) -> list[dict[str, Any]]:
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
    return _score_users(_WORKER_STATE["ranker"], user_ids, _WORKER_STATE["k"])


def _score_all(
    ranker: MatrixRanker, user_ids: Sequence[str], k: int, workers: int
) -> list[dict[str, Any]]:
    if workers <= 1 or len(user_ids) < 2:
        return _score_users(ranker, user_ids, k)
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


def _assert_completed_manifest_is_absent(path: Path) -> None:
    if not path.is_file():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return
    if isinstance(value, dict) and value.get("status") == "complete":
        raise FileExistsError(f"completed ranking manifest already exists: {path}")


def run_rankings_batch(
    manifest_path: Path,
    *,
    model_id: str,
    cutoff_name: str,
    output_path: Path,
    run_manifest_path: Path | None = None,
    parameters: BaselineParameters | None = None,
    k: int = 100,
    workers: int = 1,
    user_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit one baseline and write a JSON Lines ranking file plus a run manifest."""

    if model_id not in BASELINE_MODEL_IDS:
        raise ValueError(f"unsupported model_id: {model_id}")
    if cutoff_name not in INTERACTIONS_BY_CUTOFF:
        raise ValueError(f"unsupported cutoff_name: {cutoff_name}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    parameters = parameters or BaselineParameters()
    run_manifest_path = run_manifest_path or output_path.with_suffix(".manifest.json")
    _assert_completed_manifest_is_absent(run_manifest_path)

    manifest = _load_manifest(manifest_path)
    snapshot_id = str(manifest["snapshot_id"])
    cutoff_timestamp = manifest["cutoffs"].get(cutoff_name)
    if not cutoff_timestamp:
        raise ValueError(f"snapshot manifest is missing cutoffs.{cutoff_name}")
    interaction_name = INTERACTIONS_BY_CUTOFF[cutoff_name]
    interaction_path = _resolve_artifact(manifest_path, manifest, interaction_name)
    source_hashes = {interaction_name: _verified_hash(manifest, interaction_name, interaction_path)}

    began = time.perf_counter()
    matrix = build_interaction_matrix(
        _read_table(interaction_path, INTERACTION_COLUMNS),
        cutoff_timestamp,
        positive_threshold=parameters.positive_threshold,
    )
    matrix_ms = (time.perf_counter() - began) * 1000.0
    if user_ids is None:
        target_name = TARGETS_BY_CUTOFF[cutoff_name]
        target_path = _resolve_artifact(manifest_path, manifest, target_name)
        source_hashes[target_name] = _verified_hash(manifest, target_name, target_path)
        user_ids = evaluation_user_ids(
            matrix,
            _read_table(target_path, ("user_id", "item_id", "rating", "timestamp")),
            parameters.positive_threshold,
        )
    else:
        user_ids = sorted(set(user_ids))

    ranker = create_baseline(model_id, parameters)
    began = time.perf_counter()
    ranker.fit(None, cutoff_timestamp, snapshot_id, matrix=matrix)
    fit_ms = (time.perf_counter() - began) * 1000.0
    began = time.perf_counter()
    rows = _score_all(ranker, user_ids, k, workers)
    scoring_ms = (time.perf_counter() - began) * 1000.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "model_id": model_id,
                "snapshot_id": snapshot_id,
                "cutoff_timestamp": cutoff_timestamp,
                **row,
                "fit_time_ms": fit_ms,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    configuration = ranker.configuration()
    seed = effective_seed(model_id, parameters.seed)
    configuration_hash = stable_model_hash(
        {"model_id": model_id, "cutoff_name": cutoff_name, "k": k, **parameters.to_dict()}
    )
    run_manifest = {
        "schema_version": 1,
        "task": "batch_rankings",
        "run_id": stable_model_hash(
            {
                "snapshot_id": snapshot_id,
                "dataset_hash": manifest["dataset_hash"],
                "model_hash": ranker.model_hash,
                "configuration_hash": configuration_hash,
                "cutoff_name": cutoff_name,
            }
        )[:16],
        "model_id": model_id,
        "model_hash": ranker.model_hash,
        "snapshot_id": snapshot_id,
        "dataset_hash": manifest["dataset_hash"],
        "data_fingerprint": matrix.fingerprint,
        "catalog_hash": matrix.catalog_hash,
        "cutoff_name": cutoff_name,
        "cutoff_timestamp": cutoff_timestamp,
        "candidate_rule": CANDIDATE_RULE,
        "seed": seed,
        "k": k,
        "parameters": parameters.to_dict(),
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "source_artifact_sha256": source_hashes,
        "users": len(rows),
        "fallback_counts": dict(Counter(row["fallback_reason"] or "none" for row in rows)),
        "timing_ms": {"matrix": matrix_ms, "fit": fit_ms, "scoring": scoring_ms},
        "workers": workers,
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "status": "complete",
    }
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=BASELINE_MODEL_IDS, required=True)
    parser.add_argument("--cutoff", choices=tuple(INTERACTIONS_BY_CUTOFF), default="t1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--k", type=_positive_integer, default=100)
    parser.add_argument("--workers", type=_positive_integer, default=1)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--neighbor-limit", type=int, default=0)
    parser.add_argument("--embedding-dimensions", type=_positive_integer, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regularization", type=float, default=0.0001)
    parser.add_argument("--epochs", type=_positive_integer, default=50)
    parser.add_argument("--negative-samples", type=_positive_integer, default=1)
    parser.add_argument("--bpr-batch-size", type=_positive_integer, default=1024)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-iter", type=_positive_integer, default=100)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--ppr-batch-size", type=_positive_integer, default=64)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run_manifest = run_rankings_batch(
        args.snapshot_manifest,
        model_id=args.model,
        cutoff_name=args.cutoff,
        output_path=args.output,
        run_manifest_path=args.manifest,
        parameters=BaselineParameters(
            positive_threshold=args.positive_threshold,
            neighbor_limit=args.neighbor_limit,
            embedding_dimensions=args.embedding_dimensions,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            epochs=args.epochs,
            negative_samples=args.negative_samples,
            bpr_batch_size=args.bpr_batch_size,
            damping=args.damping,
            max_iter=args.max_iter,
            tol=args.tol,
            ppr_batch_size=args.ppr_batch_size,
            seed=args.seed,
        ),
        k=args.k,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                key: run_manifest[key]
                for key in ("run_id", "model_id", "snapshot_id", "users", "fallback_counts")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
