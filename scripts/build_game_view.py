"""Build the ADR-0022 game-only view of a snapshot.

Each cutoff uses its own ADR-0021 scope file: interactions and targets for
``t0`` drop the non-game items found with pre-``T0`` labels, and those for
``t1`` drop the items found with pre-``T1`` labels. The view removes non-game
items from histories, the candidate catalog, and targets. Other tables stay
as they are. The result is a derived snapshot manifest with its own ID,
dataset hash, and artifact hashes, so every ranking and evaluation script
runs on it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from scripts.classify_item_scope import load_item_scope
    from scripts.run_rankings_batch import (
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
    )
except ModuleNotFoundError:  # run as ``python scripts/build_game_view.py``
    from classify_item_scope import load_item_scope  # type: ignore[no-redef]
    from run_rankings_batch import (  # type: ignore[no-redef]
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
        _verified_hash,
    )

TABLES_BY_CUTOFF = {
    "t0": ("train_interactions", "validation_targets"),
    "t1": ("fit_interactions", "test_targets"),
}


def build_game_view(
    manifest_path: Path, *, scope_paths: dict[str, Path], output_dir: Path
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = json.loads(json.dumps(manifest))
    derived["artifact_paths"] = {
        name: str(_resolve_artifact(manifest_path, manifest, name).resolve())
        for name in manifest["artifact_paths"]
    }
    derived.setdefault("artifact_sha256", {})
    counts: dict[str, Any] = {}
    for cutoff, tables in TABLES_BY_CUTOFF.items():
        non_game = sorted(load_item_scope(scope_paths[cutoff]))
        for name in tables:
            source = _resolve_artifact(manifest_path, manifest, name)
            _verified_hash(manifest, name, source)
            table = pq.read_table(source)
            kept = table.filter(pc.invert(pc.is_in(table["item_id"], value_set=pa.array(non_game))))
            target = output_dir / f"{name}.parquet"
            pq.write_table(kept, target)
            derived["artifact_paths"][name] = str(target.resolve())
            derived["artifact_sha256"][name] = _sha256_file(target)
            counts[name] = {"rows": table.num_rows, "game_rows": kept.num_rows}
    digest = hashlib.sha256(
        json.dumps(derived["artifact_sha256"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    derived["snapshot_id"] = f"{manifest['snapshot_id']}-games"
    derived["dataset_hash"] = digest
    derived["derived_from"] = {
        "snapshot_id": manifest["snapshot_id"],
        "dataset_hash": manifest["dataset_hash"],
        "manifest_sha256": _sha256_file(manifest_path),
        "item_scope_sha256": {cutoff: _sha256_file(path) for cutoff, path in scope_paths.items()},
        "rule": "ADR-0022: drop non-game items (ADR-0021) from interactions and targets",
        "row_counts": counts,
    }
    derived_path = output_dir / "snapshot.json"
    derived_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(derived_path), "dataset_hash": digest, "row_counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--t0-scope", type=Path, required=True)
    parser.add_argument("--t1-scope", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_game_view(
        args.snapshot_manifest,
        scope_paths={"t0": args.t0_scope, "t1": args.t1_scope},
        output_dir=args.output_dir,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
