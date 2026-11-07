"""Explain the top-K of the ADR-0017 H0 for every evaluation user (T7.1).

The script refits the same components as ``run_hybrid_batch.py``, recomputes
each user's top-K, and checks it against the saved ranking file, so the
explanations never change the ranking (ADR-0018). Each recommendation gets
its score parts and up to ``max_claims`` aspect claims built from the curated
LLM evidence. The summary reports coverage and abstention by history slice,
item popularity, and item age. Under ADR-0010 these numbers stay provisional
until the T7.3 correctness audit.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trustrec.explanations.batch import build_claim_templates, explain_with_templates
from trustrec.explanations.claims import ClaimThresholds
from trustrec.explanations.evidence import aggregate_evidence
from trustrec.recommenders.contracts import stable_model_hash, timestamp_value
from trustrec.recommenders.hybrid_batch import HybridConfig
from trustrec.recommenders.matrix import MS_PER_DAY, top_k_order

try:
    from scripts.run_hybrid_batch import _components
    from scripts.run_rankings_batch import (
        _assert_completed_manifest_is_absent,
        _read_table,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
        evaluation_user_ids,
    )
except ModuleNotFoundError:  # run as ``python scripts/run_explanations_batch.py``
    from run_hybrid_batch import _components  # type: ignore[no-redef]
    from run_rankings_batch import (  # type: ignore[no-redef]
        _assert_completed_manifest_is_absent,
        _read_table,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
        evaluation_user_ids,
    )

EVIDENCE_FIELDS = (
    "review_id",
    "item_id",
    "author_id",
    "timestamp",
    "aspect",
    "sentiment",
    "sentiment_score",
    "confidence",
    "duplicate_group",
    "start",
    "end",
    "evidence",
)
HISTORY_SLICES = ("no_history", "1-2", "3-5", ">5")
POPULARITY_BUCKETS = ((0, "0"), (10, "1-9"), (100, "10-99"), (1000, "100-999"))
AGE_BUCKETS = ((60, "<60d"), (365, "60-364d"), (1095, "1-3y"))
THRESHOLD_GRID = [
    {"min_distinct_authors": authors, "min_support": support}
    for authors in (3, 5, 10)
    for support in (0.1, 0.25, 0.5)
]
_WORKER_STATE: dict[str, Any] = {}


def _history_slice(count: int) -> str:
    if count == 0:
        return "no_history"
    if count <= 2:
        return "1-2"
    return "3-5" if count <= 5 else ">5"


def _bucket(value: float, buckets: Sequence[tuple[float, str]], last: str) -> str:
    for upper, name in buckets:
        if value < upper:
            return name
    return last


def _score_user(user_id: str) -> dict[str, Any]:
    """Return one user's top-K item IDs, scores, and weighted score parts."""

    scorer, config, k = _WORKER_STATE["scorer"], _WORKER_STATE["config"], _WORKER_STATE["k"]
    matrix = scorer.matrix
    if scorer.cold_start == "fallback" and matrix.positive_indices(user_id).size == 0:
        recommendation, _ = scorer.recommend_many_configs(user_id, [config], k)[0]
        return {
            "user_id": user_id,
            "item_ids": list(recommendation.item_ids),
            "scores": list(recommendation.scores),
            "score_parts": [{"recent_popularity": score} for score in recommendation.scores],
            "aspect_in_score": False,
            "fallback_reason": recommendation.fallback_reason,
        }
    parts = scorer.components(user_id)
    total, _ = scorer.totals(parts, config)
    order = top_k_order(total, parts.candidates, k)
    weighted = {
        "mf": config.mf_weight * parts.mf,
        "graph": config.graph_weight * parts.graph,
        "recency": config.recency_weight * parts.recency,
        "aspect": config.aspect_weight * parts.aspect,
    }
    return {
        "user_id": user_id,
        "item_ids": [matrix.item_ids[int(parts.candidates[p])] for p in order],
        "scores": [float(total[p]) for p in order],
        "score_parts": [
            {name: float(values[p]) for name, values in weighted.items()} for p in order
        ],
        "aspect_in_score": config.aspect_weight > 0,
        "fallback_reason": None,
    }


def _score_chunk(user_ids: list[str]) -> list[dict[str, Any]]:
    scorer = _WORKER_STATE["scorer"]
    rows = []
    for start in range(0, len(user_ids), 64):
        chunk = user_ids[start : start + 64]
        scorer.prepare(chunk)
        rows.extend(_score_user(user_id) for user_id in chunk)
        scorer.clear_cache()
    return rows


def _score_users(scorer: Any, config: HybridConfig, user_ids: list[str], k: int, workers: int):
    _WORKER_STATE.update(scorer=scorer, config=config, k=k)
    try:
        if workers <= 1:
            return _score_chunk(user_ids)
        size = max(1, -(-len(user_ids) // (workers * 8)))
        chunks = [user_ids[i : i + size] for i in range(0, len(user_ids), size)]
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            return [row for part in pool.map(_score_chunk, chunks, chunksize=1) for row in part]
    finally:
        _WORKER_STATE.clear()


def _saved_top_k(ranking_path: Path, k: int) -> dict[str, list[str]]:
    saved = {}
    with ranking_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved[row["user_id"]] = row["ranked_item_ids"][:k]
    return saved


def _evidence_for(path: Path, item_ids: set[str]) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["item_id"] in item_ids:
                    yield {key: row[key] for key in EVIDENCE_FIELDS if key in row}


def _review_texts(
    manifest_path: Path, manifest: dict[str, Any], review_ids: set[str]
) -> dict[str, str]:
    path = _resolve_artifact(manifest_path, manifest, "review_texts")
    _verified_hash(manifest, "review_texts", path)
    table = pq.read_table(path, columns=["review_id", "text"])
    table = table.filter(pc.is_in(table["review_id"], value_set=pa.array(sorted(review_ids))))
    texts = {
        review_id: text or ""
        for review_id, text in zip(
            table["review_id"].to_pylist(), table["text"].to_pylist(), strict=True
        )
    }
    missing = review_ids - texts.keys()
    if missing:
        raise ValueError(f"review_texts lacks {len(missing)} cited reviews")
    return texts


def _share(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: value / total for key, value in sorted(counter.items())} if total else {}


def summarize(rows: Sequence[dict[str, Any]], templates: dict[tuple[str, str], Any]) -> dict:
    """Coverage, abstention, and refusal counts by group (provisional until T7.3)."""

    def status_table(group_key: str) -> dict[str, Any]:
        groups: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            groups[str(row[group_key])][row["status"]] += 1
        return {
            name: {"recommendations": sum(c.values()), **_share(c, sum(c.values()))}
            for name, c in sorted(groups.items())
        }

    users: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = users.setdefault(row["user_id"], {"slice": row["history_slice"], "any": False})
        entry["any"] = entry["any"] or row["status"] == "supported"
    user_cover: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in users.values():
        user_cover[entry["slice"]]["users"] += 1
        user_cover[entry["slice"]]["with_supported"] += int(entry["any"])
    refusals: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    dominant: Counter[str] = Counter()
    for row in rows:
        for claim in row["claims"]:
            if claim["status"] == "supported":
                directions[claim["sentiment_direction"]] += 1
            elif claim["refusal_reason"]:
                refusals.update(claim["refusal_reason"].split(": "))
        if row["aspect_in_score"]:
            dominant[row["dominant_score_part"]] += 1
    sensitivity = []
    for grid in THRESHOLD_GRID:
        thresholds = ClaimThresholds(**grid)
        supported = sum(
            any(
                templates[(row["item_id"], claim["aspect"])].status_under(thresholds) == "supported"
                for claim in row["claims"]
            )
            for row in rows
        )
        sensitivity.append({**grid, "supported_share": supported / len(rows)})
    statuses = Counter(row["status"] for row in rows)
    return {
        "correctness_pending": "ADR-0010: report coverage only with the T7.3 correctness audit",
        "recommendations": len(rows),
        "users": len(users),
        "status_share": _share(statuses, len(rows)),
        "status_by_aspect_in_score": status_table("aspect_in_score"),
        "status_by_history_slice": status_table("history_slice"),
        "status_by_item_popularity": status_table("popularity_bucket"),
        "status_by_item_age": status_table("age_bucket"),
        "status_by_rank": status_table("rank"),
        "users_with_supported_item": {
            name: {**dict(c), "share": c["with_supported"] / c["users"]}
            for name, c in sorted(user_cover.items())
        },
        "supported_directions": dict(sorted(directions.items())),
        "refusal_reasons": dict(refusals.most_common()),
        "dominant_score_part_when_aspect_scored": dict(dominant.most_common()),
        "threshold_sensitivity": sensitivity,
    }


def run_explanations(
    manifest_path: Path,
    *,
    evidence_path: Path,
    ranking_path: Path,
    config: HybridConfig,
    scorer_settings: dict[str, Any],
    output_dir: Path,
    cutoff_name: str = "t1",
    mf_parameters: dict[str, Any] | None = None,
    graph_parameters: dict[str, Any] | None = None,
    mf_model_path: Path | None = None,
    k: int = 10,
    max_claims: int = 1,
    thresholds: ClaimThresholds | None = None,
    workers: int = 1,
    positive_threshold: float = 4.0,
) -> dict[str, Any]:
    if config.model_id != "h0_fixed_hybrid":
        raise ValueError("T7.1 explains the ADR-0017 H0 only")
    thresholds = thresholds or ClaimThresholds()
    summary_path = output_dir / "summary.json"
    _assert_completed_manifest_is_absent(summary_path)
    began = time.perf_counter()
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
    interactions = context.pop("interactions")
    matrix, scorer = context["matrix"], context["scorer"]
    aspect_model = scorer.aspect_model
    user_ids = evaluation_user_ids(
        matrix,
        _read_table(context["target_path"], ("user_id", "item_id", "rating", "timestamp")),
        positive_threshold,
    )
    scored = _score_users(scorer, config, list(user_ids), k, workers)

    saved = _saved_top_k(ranking_path, k)
    mismatched = [row["user_id"] for row in scored if saved.get(row["user_id"]) != row["item_ids"]]
    if mismatched or len(saved) != len(scored):
        raise ValueError(
            f"recomputed top-{k} differs from {ranking_path} for {len(mismatched)} users"
        )

    item_ids = {item for row in scored for item in row["item_ids"]}
    evidence_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _evidence_for(evidence_path, item_ids):
        evidence_by_item[row["item_id"]].append(row)
    texts = _review_texts(
        manifest_path,
        context["manifest"],
        {row["review_id"] for rows in evidence_by_item.values() for row in rows},
    )
    for rows in evidence_by_item.values():
        for row in rows:
            # select_explanation checks each passage against the full review text.
            row["source_text"] = texts[row["review_id"]]
    priors = dict(zip(aspect_model.aspects, aspect_model.aspect_priors.tolist(), strict=True))
    cutoff = context["cutoff_timestamp"]
    templates: dict[tuple[str, str], Any] = {}
    max_profile_gap = 0.0
    for item_id in sorted(item_ids):
        rows = evidence_by_item.get(item_id, [])
        profiles = aggregate_evidence(
            rows,
            cutoff_timestamp=cutoff,
            prior_scores=priors,
            shrinkage_lambda=5.0,
            items=[item_id],
            aspects=aspect_model.aspects,
        )
        index = matrix.item_index[item_id]
        for position, aspect in enumerate(aspect_model.aspects):
            profile = profiles[(item_id, aspect)]
            gap = abs(
                (profile.score - profile.prior_score) - aspect_model.item_deviation[index, position]
            )
            max_profile_gap = max(max_profile_gap, gap)
        for aspect, template in build_claim_templates(
            item_id, profiles, rows, aspect_model.aspects, cutoff, thresholds
        ).items():
            templates[(item_id, aspect)] = template
    if max_profile_gap > 1e-9:
        raise ValueError(f"explanation profiles differ from ranking profiles by {max_profile_gap}")

    first_seen = interactions.group_by("item_id").aggregate([("timestamp", "min")])
    first_ms = dict(
        zip(
            first_seen["item_id"].to_pylist(),
            pc.cast(first_seen["timestamp_min"], "int64").to_pylist(),
            strict=True,
        )
    )
    cutoff_ms = timestamp_value(cutoff)
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendations_path = output_dir / "recommendations.jsonl"
    rows_out: list[dict[str, Any]] = []
    with recommendations_path.open("w", encoding="utf-8") as handle:
        for user in scored:
            user_index = matrix.user_index.get(user["user_id"])
            weights = dict(
                zip(
                    aspect_model.aspects,
                    aspect_model.user_weights(user_index).tolist(),
                    strict=True,
                )
            )
            history = int(matrix.history_indices(user["user_id"]).size)
            for rank, (item_id, score, parts) in enumerate(
                zip(user["item_ids"], user["scores"], user["score_parts"], strict=True), start=1
            ):
                item_templates = {
                    aspect: templates[(item_id, aspect)]
                    for aspect in aspect_model.aspects
                    if (item_id, aspect) in templates
                }
                explanation = explain_with_templates(
                    item_templates,
                    weights,
                    max_claims=max_claims,
                    aspect_in_score=user["aspect_in_score"],
                )
                popularity = int(matrix.popularity[matrix.item_index[item_id]])
                age_days = (cutoff_ms - first_ms[item_id]) / MS_PER_DAY
                row = {
                    "user_id": user["user_id"],
                    "rank": rank,
                    "item_id": item_id,
                    "score": score,
                    "score_parts": parts,
                    "dominant_score_part": max(parts, key=lambda name: (abs(parts[name]), name)),
                    "history_count": history,
                    "history_slice": _history_slice(history),
                    "item_popularity": popularity,
                    "popularity_bucket": _bucket(popularity, POPULARITY_BUCKETS, ">=1000"),
                    "item_age_days": age_days,
                    "age_bucket": _bucket(age_days, AGE_BUCKETS, ">=3y"),
                    **explanation,
                }
                rows_out.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    claims_path = output_dir / "claims.jsonl"
    with claims_path.open("w", encoding="utf-8") as handle:
        for key in sorted(templates):
            handle.write(json.dumps(templates[key].to_dict(), ensure_ascii=False) + "\n")

    configuration = {
        "hybrid": asdict(config),
        "scorer_settings": context["scorer_settings"],
        "k": k,
        "max_claims": max_claims,
        "thresholds": asdict(thresholds),
        "shrinkage_lambda": 5.0,
        "evidence_source": "curated LLM evidence (ADR-0018)",
    }
    summary = {
        "schema_version": 1,
        "task": "T7.1",
        "status": "complete",
        "snapshot_id": context["snapshot_id"],
        "dataset_hash": context["manifest"]["dataset_hash"],
        "cutoff_name": cutoff_name,
        "cutoff_timestamp": cutoff,
        "configuration": configuration,
        "configuration_hash": stable_model_hash(configuration),
        "component_model_hashes": context["component_hashes"],
        "source_artifact_sha256": {
            **context["source_hashes"],
            "ranking": _sha256_file(ranking_path),
        },
        "checks": {
            "ranking_matches_saved_file": True,
            "max_profile_gap_vs_ranker": max_profile_gap,
            "explained_items": len(item_ids),
        },
        "outputs": {
            "recommendations": {
                "path": str(recommendations_path),
                "sha256": _sha256_file(recommendations_path),
            },
            "claims": {"path": str(claims_path), "sha256": _sha256_file(claims_path)},
        },
        "metrics": summarize(rows_out, templates),
        "timing_ms": {"total": (time.perf_counter() - began) * 1000.0},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True, help="saved H0 ranking file")
    parser.add_argument("--h0", type=json.loads, required=True, help="H0 JSON weights")
    parser.add_argument("--cutoff", choices=("t0", "t1"), default="t1")
    parser.add_argument("--mf-model", type=Path)
    parser.add_argument("--mf-params", type=json.loads, default={})
    parser.add_argument("--graph-params", type=json.loads, default={})
    parser.add_argument("--normalization", default="minmax")
    parser.add_argument("--cold-start", default="fallback")
    parser.add_argument("--cold-window-days", type=float)
    parser.add_argument("--recency-window-days", type=float)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-claims", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_explanations(
        args.snapshot_manifest,
        evidence_path=args.evidence,
        ranking_path=args.ranking,
        config=HybridConfig("h0_fixed_hybrid", **args.h0),
        scorer_settings={
            "normalization": args.normalization,
            "cold_start": args.cold_start,
            "cold_window_days": args.cold_window_days,
            "recency_window_days": args.recency_window_days,
        },
        output_dir=args.output_dir,
        cutoff_name=args.cutoff,
        mf_parameters=args.mf_params,
        graph_parameters=args.graph_params,
        mf_model_path=args.mf_model,
        k=args.k,
        max_claims=args.max_claims,
        workers=args.workers,
    )
    metrics = summary["metrics"]
    print(json.dumps({key: metrics[key] for key in ("recommendations", "users", "status_share")}))


if __name__ == "__main__":
    main()
