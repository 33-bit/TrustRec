"""Label pre-cutoff reviews with an LLM into a resumable JSON Lines cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trustrec.data.hf_sync import HFSync
from trustrec.nlp.llm_labeler import (
    PROMPT_VERSION,
    ChatClient,
    InsufficientBalanceError,
    ReviewResult,
    ShardedCacheWriter,
    UsageTotals,
    batched,
    label_batch,
    load_cache,
    prompt_hash,
)
from trustrec.recommenders.contracts import timestamp_value

try:
    from scripts.run_rankings_batch import _load_manifest, _resolve_artifact, _verified_hash
except ModuleNotFoundError:  # run as ``python scripts/label_reviews_llm.py``
    from run_rankings_batch import (  # type: ignore[no-redef]
        _load_manifest,
        _resolve_artifact,
        _verified_hash,
    )

INTERACTIONS_BY_SCOPE = {"t0": "train_interactions", "t1": "fit_interactions"}


def _order_key(review_id: str) -> str:
    return hashlib.sha256(review_id.encode("utf-8")).hexdigest()


def select_reviews(
    manifest_path: Path, manifest: dict[str, Any], scope: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return reviews in the scope's interaction table, pre-T0 first, in hashed order."""

    interaction_name = INTERACTIONS_BY_SCOPE[scope]
    interaction_path = _resolve_artifact(manifest_path, manifest, interaction_name)
    texts_path = _resolve_artifact(manifest_path, manifest, "review_texts")
    hashes = {
        interaction_name: _verified_hash(manifest, interaction_name, interaction_path),
        "review_texts": _verified_hash(manifest, "review_texts", texts_path),
    }
    review_ids = pq.read_table(interaction_path, columns=["review_id"])["review_id"]
    cutoff_ms = int(timestamp_value(manifest["cutoffs"][scope]))
    t0_ms = timestamp_value(manifest["cutoffs"]["t0"])
    texts = pq.read_table(
        texts_path, columns=["review_id", "text", "timestamp", "source_text_sha256"]
    )
    keep = pc.and_(
        pc.is_in(texts["review_id"], value_set=pa.array(review_ids.to_pylist())),
        pc.less(texts["timestamp"], cutoff_ms),
    )
    rows = texts.filter(keep).to_pylist()
    for row in rows:
        row["text"] = row["text"] or ""
        actual = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if row["source_text_sha256"] and row["source_text_sha256"] != actual:
            raise ValueError(f"review {row['review_id']} text does not match its hash")
        row["source_text_sha256"] = actual
    rows.sort(key=lambda row: (row["timestamp"] >= t0_ms, _order_key(row["review_id"])))
    return rows, hashes


def _balance(api_key: str, base_url: str) -> float | None:
    import requests

    try:
        response = requests.get(
            f"{base_url}/user/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        infos = response.json().get("balance_infos", [])
        return sum(float(info.get("total_balance", 0.0)) for info in infos)
    except Exception:
        return None


def run_llm_labeling(
    manifest_path: Path,
    *,
    cache_path: Path,
    client: ChatClient,
    scope: str = "t1",
    batch_size: int = 20,
    workers: int = 16,
    max_reviews: int | None = None,
    max_chars: int = 4000,
    min_balance: float | None = 1.0,
    balance_every: int = 50,
    balance_reader: Any = None,
    rows_per_part: int = 50_000,
    sync: Callable[[], Any] | None = None,
    sync_every_s: float = 600.0,
) -> dict[str, Any]:
    """Label uncached reviews and append one JSON line per review to the cache.

    ``cache_path`` is a single JSON Lines file or a directory of ``part-*.jsonl``
    files. A directory suits long runs: finished parts never change, and
    ``sync`` (for example an upload to a dataset repository) runs every
    ``sync_every_s`` seconds and once at the end.
    """

    if scope not in INTERACTIONS_BY_SCOPE:
        raise ValueError(f"unsupported scope: {scope}")
    manifest = _load_manifest(manifest_path)
    reviews, source_hashes = select_reviews(manifest_path, manifest, scope)
    cached = load_cache(cache_path, client.model_id, client.reasoning_effort)
    pending = [row for row in reviews if row["review_id"] not in cached]
    if max_reviews is not None:
        pending = pending[:max_reviews]
    read_balance = balance_reader or (lambda: _balance(client.api_key, client.base_url))
    start_balance = read_balance() if min_balance is not None else None

    usage = UsageTotals()
    outcomes: Counter[str] = Counter()
    stop_reason = "complete"
    began = time.perf_counter()
    lock = threading.Lock()
    sharded = cache_path.is_dir() or not cache_path.suffix
    if sharded:
        writer: Any = ShardedCacheWriter(cache_path, rows_per_part=rows_per_part)
        runs_path = cache_path / "runs.jsonl"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cache_path.open("a", encoding="utf-8")
        runs_path = cache_path.with_suffix(".runs.jsonl")
    last_sync = time.monotonic()
    sync_results: list[Any] = []

    def work(batch: list[dict[str, Any]]) -> tuple[list[Any], dict[str, Any]]:
        return label_batch(client, batch, max_chars)

    def batches() -> Iterator[list[dict[str, Any]]]:
        empty = [row for row in pending if not row["text"].strip()]
        if empty:
            yield empty
        yield from batched((row for row in pending if row["text"].strip()), batch_size)

    with ThreadPoolExecutor(workers) as pool:
        in_flight: dict[Future[Any], tuple[str, list[dict[str, Any]]]] = {}
        source = batches()
        stop = False

        def write(results: list[Any], request_id: str) -> None:
            with lock:
                for result in results:
                    writer.write(
                        json.dumps(
                            result.to_record(client.model_id, request_id, client.reasoning_effort),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    outcomes[result.error or ("labeled" if result.labels else "no_labels")] += 1
                writer.flush()

        while True:
            while not stop and len(in_flight) < workers * 2:
                batch = next(source, None)
                if batch is None:
                    break
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
                request_id = f"{stamp}-{_order_key(batch[0]['review_id'])[:8]}"
                if not batch[0]["text"].strip():
                    write(
                        [
                            ReviewResult(
                                review_id=row["review_id"],
                                source_text_sha256=row["source_text_sha256"],
                                out_of_scope=False,
                                needs_review=True,
                                labels=[],
                            )
                            for row in batch
                        ],
                        "empty-text",
                    )
                    continue
                in_flight[pool.submit(work, batch)] = (request_id, batch)
            if not in_flight:
                break
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                request_id, batch = in_flight.pop(future)
                try:
                    results, request_usage = future.result()
                except InsufficientBalanceError:
                    stop, stop_reason = True, "insufficient_balance"
                    continue
                except RuntimeError as error:
                    usage.failed_requests += 1
                    usage.errors[str(error)[:80]] = usage.errors.get(str(error)[:80], 0) + 1
                    continue
                usage.add(request_usage, len(batch))
                write(results, request_id)
                if min_balance is not None and usage.requests % balance_every == 0:
                    balance = read_balance()
                    if balance is not None and balance < min_balance:
                        stop, stop_reason = True, "min_balance_reached"
                if sync is not None and time.monotonic() - last_sync >= sync_every_s:
                    with lock:
                        writer.flush()
                    try:
                        sync_results.append(sync())
                    except Exception as error:  # keep labeling when an upload fails
                        sync_results.append({"error": str(error)[:200]})
                    last_sync = time.monotonic()
                if usage.requests % 100 == 0:
                    progress = {
                        "requests": usage.requests,
                        "reviews": usage.reviews,
                        "pending": len(pending) - usage.reviews,
                        "elapsed_s": round(time.perf_counter() - began),
                    }
                    print(json.dumps(progress), flush=True)
    remaining = len(reviews) - len(cached)
    if stop_reason == "complete" and max_reviews is not None and remaining > max_reviews:
        stop_reason = "max_reviews_reached"

    writer.close()
    final_cache = load_cache(cache_path, client.model_id, client.reasoning_effort)
    end_balance = read_balance() if min_balance is not None else None
    run = {
        "schema_version": 1,
        "task": "T6.2-llm",
        "snapshot_id": manifest["snapshot_id"],
        "dataset_hash": manifest["dataset_hash"],
        "scope": scope,
        "source_cutoff_timestamp": manifest["cutoffs"][scope],
        "source_artifact_sha256": source_hashes,
        "model_id": client.model_id,
        "model_revision": "unavailable",
        "provider_base_url": client.base_url,
        "reasoning_effort": client.reasoning_effort,
        "temperature": 0,
        "seed": "unavailable",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "label_status": "llm_silver",
        "batch_size": batch_size,
        "max_chars": max_chars,
        "scope_reviews": len(reviews),
        "cached_reviews": len(final_cache),
        "coverage": len(final_cache) / len(reviews) if reviews else 1.0,
        "this_run": {
            "usage": usage.to_dict(),
            "outcomes": dict(outcomes),
            "elapsed_s": time.perf_counter() - began,
            "start_balance": start_balance,
            "end_balance": end_balance,
            "stop_reason": stop_reason,
        },
        "finished_at": datetime.now(UTC).isoformat(),
        "status": "complete" if len(final_cache) == len(reviews) else "partial",
    }
    with runs_path.open("a", encoding="utf-8") as runs:
        runs.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
    if sync is not None:
        try:
            sync_results.append(sync())
        except Exception as error:
            sync_results.append({"error": str(error)[:200]})
        run["sync_results"] = sync_results[-3:]
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(INTERACTIONS_BY_SCOPE), default="t1")
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-flash"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-reviews", type=int)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--min-balance", type=float, default=1.0)
    parser.add_argument("--rows-per-part", type=int, default=50_000)
    parser.add_argument("--hf-repo", help="dataset repository to mirror the cache directory")
    parser.add_argument("--hf-prefix", help="folder inside the repository, for example llm/<model>")
    parser.add_argument("--hf-token-env", default="HF_KEY")
    parser.add_argument("--hf-allow-public", action="store_true")
    parser.add_argument("--sync-minutes", type=float, default=10.0)
    args = parser.parse_args()
    sync = None
    if args.hf_repo:
        if not args.hf_prefix:
            raise SystemExit("--hf-prefix is required with --hf-repo")
        hf = HFSync(
            repo_id=args.hf_repo,
            local_dir=args.cache,
            prefix=args.hf_prefix,
            token_env=args.hf_token_env,
            allow_public=args.hf_allow_public,
        )
        hf.check_visibility()
        print(json.dumps({"hf_pull": hf.pull()}), flush=True)
        sync = hf.push
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} to the provider API key")
    run = run_llm_labeling(
        args.snapshot_manifest,
        cache_path=args.cache,
        client=ChatClient(
            api_key=api_key,
            model_id=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort or None,
        ),
        scope=args.scope,
        batch_size=args.batch_size,
        workers=args.workers,
        max_reviews=args.max_reviews,
        max_chars=args.max_chars,
        min_balance=None if args.min_balance < 0 else args.min_balance,
        rows_per_part=args.rows_per_part,
        sync=sync,
        sync_every_s=args.sync_minutes * 60.0,
    )
    summary = {key: run[key] for key in ("cached_reviews", "scope_reviews", "coverage", "status")}
    print(json.dumps({**summary, "this_run": run["this_run"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
