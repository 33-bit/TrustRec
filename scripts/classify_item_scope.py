"""Write the ADR-0021 game / non-game classification for one snapshot cutoff."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trustrec.data.item_scope import classify_items, rules
from trustrec.nlp.llm_curation import read_curated_manifest

try:
    from scripts.run_rankings_batch import (
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
    )
except ModuleNotFoundError:  # run as ``python scripts/classify_item_scope.py``
    from run_rankings_batch import (  # type: ignore[no-redef]
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
    )

INTERACTIONS_BY_CUTOFF = {"t0": "train_interactions", "t1": "fit_interactions"}


def classify_item_scope(
    manifest_path: Path, *, cutoff_name: str, curated_dir: Path, output_path: Path
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    metadata_path = _resolve_artifact(manifest_path, manifest, "item_metadata")
    interactions_path = _resolve_artifact(
        manifest_path, manifest, INTERACTIONS_BY_CUTOFF[cutoff_name]
    )
    curated = read_curated_manifest(curated_dir)
    titles_table = pq.read_table(metadata_path, columns=["item_id", "title"])
    titles = dict(
        zip(titles_table["item_id"].to_pylist(), titles_table["title"].to_pylist(), strict=True)
    )
    interactions = pq.read_table(interactions_path, columns=["review_id", "item_id"])
    item_of = dict(
        zip(
            interactions["review_id"].to_pylist(),
            interactions["item_id"].to_pylist(),
            strict=True,
        )
    )
    reviews = pq.read_table(curated_dir / "reviews.parquet", columns=["review_id", "status"])
    labeled: Counter[str] = Counter()
    out_of_scope: Counter[str] = Counter()
    for review_id, status in zip(
        reviews["review_id"].to_pylist(), reviews["status"].to_pylist(), strict=True
    ):
        item_id = item_of.get(review_id)
        if item_id is None:
            continue
        labeled[item_id] += 1
        out_of_scope[item_id] += int(status == "out_of_scope")
    counts = {item_id: (labeled[item_id], out_of_scope[item_id]) for item_id in labeled}
    classes = classify_items(titles, counts)
    non_game = {item_id: cls for item_id, cls in sorted(classes.items()) if cls != "game"}
    result = {
        "schema_version": 1,
        "task": "ADR-0021 item scope",
        "snapshot_id": manifest["snapshot_id"],
        "dataset_hash": manifest["dataset_hash"],
        "cutoff_name": cutoff_name,
        "rules": rules(),
        "label_source": {
            "model_id": curated["model_id"],
            "cleaning_version": curated["cleaning_version"],
            "manifest_sha256": _sha256_file(curated_dir / "manifest.json"),
        },
        "source_artifact_sha256": {
            "item_metadata": _verified_hash(manifest, "item_metadata", metadata_path),
            INTERACTIONS_BY_CUTOFF[cutoff_name]: _verified_hash(
                manifest, INTERACTIONS_BY_CUTOFF[cutoff_name], interactions_path
            ),
        },
        "counts": dict(sorted(Counter(classes.values()).items())),
        "non_game_items": non_game,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def load_item_scope(path: Path) -> dict[str, str]:
    """Return the non-game items (item ID to class) from a scope file."""

    return dict(json.loads(path.read_text(encoding="utf-8"))["non_game_items"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--cutoff", choices=tuple(INTERACTIONS_BY_CUTOFF), default="t1")
    parser.add_argument("--curated", type=Path, required=True, help="curated LLM label folder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify_item_scope(
        args.snapshot_manifest,
        cutoff_name=args.cutoff,
        curated_dir=args.curated,
        output_path=args.output,
    )
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
