"""Audit a stratified sample of explanation claims with an LLM judge (T7.3).

The script samples supported (item, aspect) claims from a T7.1 run, checks
each cited passage against the full review text, and asks a judge model to
rate aspect, polarity, product, and claim support. Use a judge model that
differs from the labeler. Results are LLM agreement, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trustrec.explanations.judge import (
    JUDGE_PROMPT,
    JUDGE_PROMPT_VERSION,
    judge_input,
    judge_prompt_hash,
    parse_judgment,
    sample_claims,
    summarize_judgments,
)
from trustrec.nlp.llm_labeler import ChatClient, InsufficientBalanceError

try:
    from scripts.run_rankings_batch import (
        _assert_completed_manifest_is_absent,
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
    )
except ModuleNotFoundError:  # run as ``python scripts/run_explanation_judge.py``
    from run_rankings_batch import (  # type: ignore[no-redef]
        _assert_completed_manifest_is_absent,
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _traceability(manifest_path: Path, sample: list[dict[str, Any]]) -> dict[str, int]:
    manifest = _load_manifest(manifest_path)
    texts_path = _resolve_artifact(manifest_path, manifest, "review_texts")
    ids = sorted({p["review_id"] for claim in sample for p in claim["cited_evidence"]})
    table = pq.read_table(texts_path, columns=["review_id", "text"])
    table = table.filter(pc.is_in(table["review_id"], value_set=pa.array(ids)))
    texts = dict(zip(table["review_id"].to_pylist(), table["text"].to_pylist(), strict=True))
    checked = matched = 0
    for claim in sample:
        for passage in claim["cited_evidence"]:
            checked += 1
            source = texts.get(passage["review_id"]) or ""
            ok = source[passage["start"] : passage["end"]] == passage["text"]
            passage["traceable"] = ok
            matched += int(ok)
    return {"passages": checked, "traceable": matched}


def run_judge(
    manifest_path: Path,
    *,
    explanations_dir: Path,
    output_path: Path,
    client: ChatClient,
    per_stratum: int = 100,
    seed: int = 7,
    workers: int = 8,
    top_exposure: int = 0,
) -> dict[str, Any]:
    _assert_completed_manifest_is_absent(output_path)
    recommendations_path = explanations_dir / "recommendations.jsonl"
    claims_path = explanations_dir / "claims.jsonl"
    templates = {(row["item_id"], row["aspect"]): row for row in _jsonl(claims_path)}
    sample = sample_claims(
        _jsonl(recommendations_path),
        templates,
        per_stratum=per_stratum,
        seed=seed,
        top_exposure=top_exposure,
    )
    trace = _traceability(manifest_path, sample)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "failures": 0}
    lock = threading.Lock()

    def judge(claim: dict[str, Any]) -> dict[str, Any]:
        last = "no attempt"
        for _ in range(2):
            try:
                content, used = client.complete(judge_input(claim))
                with lock:
                    usage["requests"] += 1
                    usage["prompt_tokens"] += int(used.get("prompt_tokens", 0) or 0)
                    usage["completion_tokens"] += int(used.get("completion_tokens", 0) or 0)
                return {**claim, "judgment": parse_judgment(content, len(claim["cited_evidence"]))}
            except InsufficientBalanceError:
                raise
            except (ValueError, RuntimeError) as error:
                last = str(error)
        with lock:
            usage["failures"] += 1
        return {**claim, "judgment": None, "judge_error": last}

    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(judge, sample))
    result = {
        "schema_version": 1,
        "task": "T7.3",
        "status": "complete",
        "result_kind": "LLM agreement, not human judgment or ground truth",
        "judge": {
            "model_id": client.model_id,
            "model_revision": "unavailable",
            "base_url": client.base_url,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "prompt_sha256": judge_prompt_hash(),
            "prompt": JUDGE_PROMPT,
            "temperature": 0,
            "seed": seed,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "source_artifact_sha256": {
            "recommendations": _sha256_file(recommendations_path),
            "claims": _sha256_file(claims_path),
        },
        "sample": {
            "per_stratum": per_stratum,
            "top_exposure": top_exposure,
            "seed": seed,
            "claims": len(rows),
        },
        "traceability": trace,
        "summary": summarize_judgments(rows),
        "usage": usage,
        "timing_ms": (time.perf_counter() - began) * 1000.0,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--explanations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="judge model; not the labeler model")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--top-exposure", type=int, default=0, help="also judge the N most shown claims"
    )
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} to the judge provider API key")
    result = run_judge(
        args.snapshot_manifest,
        explanations_dir=args.explanations_dir,
        output_path=args.output,
        client=ChatClient(
            api_key=api_key,
            model_id=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort or None,
            system_prompt=JUDGE_PROMPT,
        ),
        per_stratum=args.per_stratum,
        seed=args.seed,
        workers=args.workers,
        top_exposure=args.top_exposure,
    )
    print(json.dumps({"summary": result["summary"], "usage": result["usage"]}))


if __name__ == "__main__":
    main()
