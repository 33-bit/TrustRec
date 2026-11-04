"""Consolidate a raw LLM label cache into clean Parquet tables and optionally upload them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trustrec.data.hf_sync import HFSync
from trustrec.nlp.llm_curation import (
    CurationStats,
    collapse_cache,
    curate_records,
    sha256_file,
    write_curated,
)
from trustrec.nlp.llm_labeler import cache_files

try:
    from scripts.label_reviews_llm import select_reviews
    from scripts.run_rankings_batch import _load_manifest
except ModuleNotFoundError:  # run as ``python scripts/curate_llm_labels.py``
    from label_reviews_llm import select_reviews  # type: ignore[no-redef]
    from run_rankings_batch import _load_manifest  # type: ignore[no-redef]

CURATED_PATTERNS = ("*.parquet", "manifest.json")


def curate_llm_labels(
    manifest_path: Path,
    *,
    cache_path: Path,
    model_id: str,
    reasoning_effort: str | None,
    output_dir: Path,
    scope: str = "t1",
) -> dict[str, Any]:
    """Curate one model's cache for one scope and return the written manifest."""

    parts = cache_files(cache_path)
    if not parts:
        raise FileNotFoundError(f"no label cache files under {cache_path}")
    manifest = _load_manifest(manifest_path)
    rows, source_hashes = select_reviews(manifest_path, manifest, scope)
    texts = {row["review_id"]: row["text"] for row in rows}
    stats = CurationStats()
    records = collapse_cache(cache_path, model_id, reasoning_effort, stats)
    review_rows, label_rows = curate_records(records, texts, stats)
    coverage = len(review_rows) / len(texts) if texts else 0.0
    return write_curated(
        output_dir,
        review_rows,
        label_rows,
        {
            "task": "T6.2-llm-curated",
            "label_status": "llm_silver",
            "model_id": model_id,
            "reasoning_effort": reasoning_effort,
            "snapshot_id": manifest["snapshot_id"],
            "dataset_hash": manifest["dataset_hash"],
            "scope": scope,
            "source_cutoff_timestamp": manifest["cutoffs"][scope],
            "source_artifact_sha256": source_hashes,
            "scope_reviews": len(texts),
            "coverage": coverage,
            "status": "complete" if len(review_rows) == len(texts) else "partial",
            "raw_cache": {
                "files": {part.name: sha256_file(part) for part in parts},
                **stats.to_dict(),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="raw cache file or directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="", help="empty string means none recorded")
    parser.add_argument("--scope", choices=("t0", "t1"), default="t1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-repo", help="dataset repository to upload the curated tables to")
    parser.add_argument("--hf-prefix", help="folder inside the repository, for example curated/<m>")
    parser.add_argument("--hf-token-env", default="HF_KEY")
    parser.add_argument("--hf-allow-public", action="store_true")
    args = parser.parse_args()
    if args.hf_repo and not args.hf_prefix:
        raise SystemExit("--hf-prefix is required with --hf-repo")
    curated = curate_llm_labels(
        args.snapshot_manifest,
        cache_path=args.cache,
        model_id=args.model,
        reasoning_effort=args.reasoning_effort or None,
        output_dir=args.output,
        scope=args.scope,
    )
    result: dict[str, Any] = {
        key: curated[key] for key in ("status", "coverage", "scope_reviews", "counts")
    }
    result["raw_cache"] = {k: v for k, v in curated["raw_cache"].items() if k != "files"}
    if args.hf_repo:
        result["hf_push"] = HFSync(
            repo_id=args.hf_repo,
            local_dir=args.output,
            prefix=args.hf_prefix,
            token_env=args.hf_token_env,
            allow_public=args.hf_allow_public,
            patterns=CURATED_PATTERNS,
        ).push()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
