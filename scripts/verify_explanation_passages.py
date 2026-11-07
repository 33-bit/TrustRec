"""Verify display passages of explanation claims with an LLM (ADR-0020).

Reads a T7.1 explanations folder and writes a new folder in the same format.
Every displayed claim shows only passages the verifier accepts. A claim with
no verified passage on its side becomes ``insufficient_support``. Scores,
rankings, and claim directions do not change. Verdicts are cached in JSON
Lines, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustrec.explanations.verify import (
    VERIFY_PROMPT,
    VERIFY_PROMPT_VERSION,
    parse_verdicts,
    passage_key,
    select_verified,
    verify_input,
    verify_prompt_hash,
)
from trustrec.nlp.llm_labeler import ChatClient, InsufficientBalanceError

try:
    from scripts.run_rankings_batch import _assert_completed_manifest_is_absent, _sha256_file
except ModuleNotFoundError:  # run as ``python scripts/verify_explanation_passages.py``
    from run_rankings_batch import (  # type: ignore[no-redef]
        _assert_completed_manifest_is_absent,
        _sha256_file,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_cache(path: Path, model_id: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for row in _jsonl(path):
            if (
                row.get("model_id") == model_id
                and row.get("prompt_version") == VERIFY_PROMPT_VERSION
            ):
                cache[row["key"]] = row["verdict"]
    return cache


def verify_explanations(
    explanations_dir: Path,
    output_dir: Path,
    *,
    client: ChatClient,
    cache_path: Path,
    workers: int = 8,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    _assert_completed_manifest_is_absent(summary_path)
    claims = _jsonl(explanations_dir / "claims.jsonl")
    recommendations = _jsonl(explanations_dir / "recommendations.jsonl")
    templates = {(row["item_id"], row["aspect"]): row for row in claims}
    shown = Counter(
        (row["item_id"], claim["aspect"])
        for row in recommendations
        for claim in row["claims"]
        if claim["status"] == "supported"
    )
    cache = _load_cache(cache_path, client.model_id)
    cached_before = len(cache)
    usage = Counter()
    lock = threading.Lock()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_handle = cache_path.open("a", encoding="utf-8")

    def verify(key: tuple[str, str]) -> None:
        claim = templates[key]
        aspect = claim["aspect"]
        missing = [p for p in claim["candidate_evidence"] if passage_key(aspect, p) not in cache]
        if not missing:
            return
        for _ in range(3):
            try:
                content, used = client.complete(verify_input(aspect, missing))
                verdicts = parse_verdicts(content, len(missing))
                break
            except InsufficientBalanceError:
                raise
            except (ValueError, RuntimeError):
                with lock:
                    usage["retries"] += 1
        else:
            with lock:
                usage["failed_claims"] += 1
            return
        with lock:
            usage["requests"] += 1
            usage["prompt_tokens"] += int(used.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(used.get("completion_tokens", 0) or 0)
            for passage, verdict in zip(missing, verdicts, strict=True):
                key_text = passage_key(aspect, passage)
                cache[key_text] = verdict
                record = {
                    "key": key_text,
                    "model_id": client.model_id,
                    "prompt_version": VERIFY_PROMPT_VERSION,
                    "verdict": verdict,
                }
                cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache_handle.flush()

    began = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(verify, sorted(shown)))
    finally:
        cache_handle.close()
    if usage["failed_claims"]:
        raise RuntimeError(f"{usage['failed_claims']} claims failed verification; rerun to resume")

    outcome: dict[tuple[str, str], bool] = {}
    new_claims = []
    for row in claims:
        key = (row["item_id"], row["aspect"])
        if key in shown:
            passages, kept = select_verified(row, cache)
            outcome[key] = kept
            row = {
                **row,
                "cited_evidence": passages,
                "display_verified": True,
                "status": "supported" if kept else "insufficient_support",
                "refusal_reason": None if kept else "no_verified_passage",
            }
        new_claims.append(row)

    for row in recommendations:
        for claim in row["claims"]:
            key = (row["item_id"], claim["aspect"])
            if claim["status"] == "supported" and not outcome.get(key, True):
                claim["status"] = "insufficient_support"
                claim["refusal_reason"] = "no_verified_passage"
        statuses = {claim["status"] for claim in row["claims"]}
        row["status"] = (
            "supported"
            if "supported" in statuses
            else ("insufficient_support" if row["claims"] else "unavailable")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("claims.jsonl", new_claims), ("recommendations.jsonl", recommendations)):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_summary = explanations_dir / "summary.json"
    if source_summary.is_file():
        shutil.copyfile(source_summary, output_dir / "source_summary.json")
    statuses = Counter(row["status"] for row in recommendations)
    kept_views = sum(count for key, count in shown.items() if outcome[key])
    summary = {
        "schema_version": 1,
        "task": "ADR-0020 display verification",
        "status": "complete",
        "verifier": {
            "model_id": client.model_id,
            "base_url": client.base_url,
            "prompt_version": VERIFY_PROMPT_VERSION,
            "prompt_sha256": verify_prompt_hash(),
            "prompt": VERIFY_PROMPT,
            "temperature": 0,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "source_sha256": {
            "claims": _sha256_file(explanations_dir / "claims.jsonl"),
            "recommendations": _sha256_file(explanations_dir / "recommendations.jsonl"),
        },
        "displayed_claims": len(shown),
        "displayed_claims_kept": sum(outcome.values()),
        "views": sum(shown.values()),
        "views_kept": kept_views,
        "recommendation_status_share": {
            name: count / len(recommendations) for name, count in sorted(statuses.items())
        },
        "kept_by_direction": dict(
            sorted(
                Counter(
                    f"{templates[key]['sentiment_direction']}:{'kept' if kept else 'dropped'}"
                    for key, kept in outcome.items()
                ).items()
            )
        ),
        "cache": {"before": cached_before, "after": len(cache)},
        "usage": dict(usage),
        "timing_ms": (time.perf_counter() - began) * 1000.0,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explanations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} to the verifier API key")
    summary = verify_explanations(
        args.explanations_dir,
        args.output_dir,
        client=ChatClient(
            api_key=api_key,
            model_id=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort or None,
            system_prompt=VERIFY_PROMPT,
        ),
        cache_path=args.cache,
        workers=args.workers,
    )
    keys = ("displayed_claims", "displayed_claims_kept", "views", "views_kept", "usage")
    print(json.dumps({key: summary[key] for key in keys}))


if __name__ == "__main__":
    main()
