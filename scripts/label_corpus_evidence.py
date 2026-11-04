"""Label aspect evidence for every review before one snapshot cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trustrec.nlp.corpus import (
    DICTIONARY_LABELER_VERSION,
    LabelingSummary,
    dictionary_fingerprint,
    label_reviews,
)
from trustrec.nlp.dictionary import DictionaryBaseline
from trustrec.nlp.llm_curation import (
    MANIFEST_FILE,
    is_curated,
    load_curated,
    read_curated_manifest,
    sha256_file,
)
from trustrec.nlp.llm_labeler import (
    PROMPT_VERSION,
    CachedLLMLabeler,
    cache_files,
    load_cache,
    prompt_hash,
)
from trustrec.recommenders.contracts import stable_model_hash, timestamp_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}
TEXT_COLUMNS = (
    "review_id",
    "item_id",
    "text",
    "timestamp",
    "source_text_sha256",
    "duplicate_group",
)
LABELERS = ("dictionary", "llm-cache")
LABEL_STATUS = {"dictionary": "rule_baseline", "llm-cache": "llm_silver"}

# A forked worker reads the active labeler from this module global.
_LABELER_STATE: dict[str, Any] = {}
USAGE = (
    "Evidence construction for item aspect profiles before the recorded cutoff. "
    "Not a human gold label, not an LLM label, and not for NLP evaluation."
)


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
    for name in ("snapshot_id", "dataset_hash", "cutoffs", "artifact_paths"):
        if not manifest.get(name):
            raise ValueError(f"snapshot manifest is missing {name}")
    return manifest


def _resolve(manifest_path: Path, manifest: dict[str, Any], name: str) -> Path:
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


def _review_batches(
    texts_path: Path,
    authors: dict[str, tuple[str, str]],
    cutoff_ms: float,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Yield pre-cutoff review texts that belong to the cutoff's interaction table."""

    review_ids = pa.array(sorted(authors), type=pa.string())
    available = set(pq.read_schema(texts_path).names)
    missing = {"review_id", "item_id", "text", "timestamp"} - available
    if missing:
        raise ValueError(f"review_texts is missing columns: {sorted(missing)}")
    columns = [name for name in TEXT_COLUMNS if name in available]
    parquet = pq.ParquetFile(texts_path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        table = pa.Table.from_batches([batch])
        keep = pc.and_(
            pc.is_in(table["review_id"], value_set=review_ids),
            pc.less(table["timestamp"], pa.scalar(int(cutoff_ms), type=pa.int64())),
        )
        rows = table.filter(keep).to_pylist()
        for row in rows:
            user_id, item_id = authors[row["review_id"]]
            if row["item_id"] != item_id:
                raise ValueError(f"review {row['review_id']} item does not match interactions")
            row["author_id"] = user_id
        if rows:
            yield rows


def _label_batch(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], LabelingSummary]:
    labeler = _LABELER_STATE.get("labeler") or DictionaryBaseline()
    source = _LABELER_STATE.get("label_source", DICTIONARY_LABELER_VERSION)
    return label_reviews(rows, labeler, label_source=source)


def _assert_completed_manifest_is_absent(path: Path) -> None:
    if not path.is_file():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return
    if isinstance(value, dict) and value.get("status") == "complete":
        raise FileExistsError(f"completed evidence manifest already exists: {path}")


def label_corpus(
    manifest_path: Path,
    *,
    cutoff_name: str,
    output_path: Path,
    run_manifest_path: Path | None = None,
    labeler: str = "dictionary",
    label_cache_path: Path | None = None,
    llm_model_id: str | None = None,
    llm_reasoning_effort: str | None = "none",
    workers: int = 1,
    batch_size: int = 20_000,
) -> dict[str, Any]:
    """Write evidence JSON Lines for one cutoff and a run manifest beside it."""

    if cutoff_name not in INTERACTIONS_BY_CUTOFF:
        raise ValueError(f"unsupported cutoff_name: {cutoff_name}")
    if labeler not in LABELERS:
        raise ValueError(f"unsupported labeler: {labeler}")
    if labeler == "llm-cache":
        if label_cache_path is None:
            raise ValueError("llm-cache needs label_cache_path")
        if is_curated(label_cache_path):
            curated = read_curated_manifest(label_cache_path)
            if llm_model_id and llm_model_id != curated["model_id"]:
                raise ValueError(
                    f"curated labels are from {curated['model_id']}, not {llm_model_id}"
                )
            llm_model_id = curated["model_id"]
            llm_reasoning_effort = curated["reasoning_effort"]
            records = load_curated(label_cache_path)
            cache_sha256 = sha256_file(label_cache_path / MANIFEST_FILE)
            cleaning = f":{curated['cleaning_version']}"
        else:
            if not llm_model_id:
                raise ValueError("a raw llm-cache needs llm_model_id")
            records = load_cache(label_cache_path, llm_model_id, llm_reasoning_effort)
            cache_sha256 = stable_model_hash(
                {part.name: sha256_file(part) for part in cache_files(label_cache_path)}
            )
            cleaning = ""
        label_source = f"{llm_model_id}:{PROMPT_VERSION}:{llm_reasoning_effort}{cleaning}"
        labeler_hash = stable_model_hash(
            {
                "label_source": label_source,
                "prompt_sha256": prompt_hash(),
                "cache_sha256": cache_sha256,
            }
        )
        _LABELER_STATE.update(labeler=CachedLLMLabeler(records), label_source=label_source)
    else:
        label_source = DICTIONARY_LABELER_VERSION
        labeler_hash = dictionary_fingerprint()
        _LABELER_STATE.clear()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    run_manifest_path = run_manifest_path or output_path.with_suffix(".manifest.json")
    _assert_completed_manifest_is_absent(run_manifest_path)

    manifest = _load_manifest(manifest_path)
    cutoff_timestamp = manifest["cutoffs"].get(cutoff_name)
    if not cutoff_timestamp:
        raise ValueError(f"snapshot manifest is missing cutoffs.{cutoff_name}")
    cutoff_ms = timestamp_value(cutoff_timestamp)
    interaction_name = INTERACTIONS_BY_CUTOFF[cutoff_name]
    interaction_path = _resolve(manifest_path, manifest, interaction_name)
    texts_path = _resolve(manifest_path, manifest, "review_texts")
    source_hashes = {
        interaction_name: _verified_hash(manifest, interaction_name, interaction_path),
        "review_texts": _verified_hash(manifest, "review_texts", texts_path),
    }
    interactions = pq.read_table(
        interaction_path, columns=["review_id", "user_id", "item_id", "timestamp"]
    )
    if pc.any(pc.greater_equal(interactions["timestamp"], int(cutoff_ms))).as_py():
        raise ValueError(f"{interaction_name} contains rows at or after the cutoff")
    authors = {
        review_id: (user_id, item_id)
        for review_id, user_id, item_id in zip(
            interactions["review_id"].to_pylist(),
            interactions["user_id"].to_pylist(),
            interactions["item_id"].to_pylist(),
            strict=True,
        )
    }

    began = time.perf_counter()
    summary = LabelingSummary()
    batches = _review_batches(texts_path, authors, cutoff_ms, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                results = pool.imap(_label_batch, batches, chunksize=1)
                for rows, part in results:
                    summary.update(part)
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            for batch in batches:
                rows, part = _label_batch(batch)
                summary.update(part)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    elapsed_ms = (time.perf_counter() - began) * 1000.0

    configuration = {"labeler": labeler, "label_source": label_source}
    run_manifest = {
        "schema_version": 1,
        "task": "T6.2",
        "run_id": stable_model_hash(
            {
                "snapshot_id": manifest["snapshot_id"],
                "dataset_hash": manifest["dataset_hash"],
                "cutoff_name": cutoff_name,
                "labeler_hash": labeler_hash,
            }
        )[:16],
        "snapshot_id": manifest["snapshot_id"],
        "dataset_hash": manifest["dataset_hash"],
        "cutoff_name": cutoff_name,
        "source_cutoff_timestamp": cutoff_timestamp,
        "label_source": label_source,
        "label_status": LABEL_STATUS[labeler],
        "usage": USAGE,
        "configuration": configuration,
        "configuration_hash": stable_model_hash(configuration),
        "model_hash": labeler_hash,
        "source_artifact_sha256": source_hashes,
        "interaction_reviews": len(authors),
        "summary": summary.to_dict(),
        "timing_ms": {"labeling": elapsed_ms},
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
    parser.add_argument("--cutoff", choices=tuple(INTERACTIONS_BY_CUTOFF), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labeler", choices=LABELERS, default="dictionary")
    parser.add_argument(
        "--label-cache",
        type=Path,
        help="raw LLM cache or curated label folder for --labeler llm-cache",
    )
    parser.add_argument(
        "--llm-model", help="model of a raw cache; defaults to MODEL, read from a curated folder"
    )
    parser.add_argument("--llm-reasoning-effort", default="none")
    parser.add_argument("--workers", type=_positive_integer, default=1)
    parser.add_argument("--batch-size", type=_positive_integer, default=20_000)
    args = parser.parse_args()
    llm_model = args.llm_model
    if llm_model is None and not (args.label_cache and is_curated(args.label_cache)):
        llm_model = os.environ.get("MODEL")
    run_manifest = label_corpus(
        args.snapshot_manifest,
        cutoff_name=args.cutoff,
        output_path=args.output,
        run_manifest_path=args.manifest,
        labeler=args.labeler,
        label_cache_path=args.label_cache,
        llm_model_id=llm_model,
        llm_reasoning_effort=args.llm_reasoning_effort,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "run_id": run_manifest["run_id"],
                "snapshot_id": run_manifest["snapshot_id"],
                "cutoff_name": run_manifest["cutoff_name"],
                **run_manifest["summary"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
