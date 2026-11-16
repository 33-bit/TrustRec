"""Build the ADR-0021 game-target slice for a secondary accuracy analysis.

The slice keeps only test targets on game items. Candidate sets, histories,
and rankings do not change, so models that recommend store credit or hardware
are not rewarded for those hits. Rankings are filtered to the users who still
have a target. The output is a derived snapshot manifest that names the new
target file and its hash. The source manifest is not changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from scripts.classify_item_scope import load_item_scope
    from scripts.run_rankings_batch import _load_manifest, _resolve_artifact, _sha256_file
except ModuleNotFoundError:  # run as ``python scripts/build_game_target_slice.py``
    from classify_item_scope import load_item_scope  # type: ignore[no-redef]
    from run_rankings_batch import (  # type: ignore[no-redef]
        _load_manifest,
        _resolve_artifact,
        _sha256_file,
    )


def build_slice(
    manifest_path: Path,
    *,
    item_scope_path: Path,
    output_dir: Path,
    rankings: dict[str, Path],
    target_name: str = "test_targets",
    positive_threshold: float = 4.0,
) -> dict[str, object]:
    manifest = _load_manifest(manifest_path)
    non_game = load_item_scope(item_scope_path)
    targets = pq.read_table(_resolve_artifact(manifest_path, manifest, target_name))
    is_game = pc.invert(pc.is_in(targets["item_id"], value_set=pa.array(sorted(non_game))))
    kept = targets.filter(is_game)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{target_name}_game_only.parquet"
    pq.write_table(kept, target_path)
    positive = kept.filter(pc.greater_equal(kept["rating"], positive_threshold))
    users = set(positive["user_id"].to_pylist())

    derived = json.loads(json.dumps(manifest))
    derived["snapshot_id"] = manifest["snapshot_id"]
    derived["artifact_paths"] = {
        name: str(_resolve_artifact(manifest_path, manifest, name).resolve())
        for name in manifest["artifact_paths"]
    }
    derived["artifact_paths"][target_name] = str(target_path.resolve())
    derived.setdefault("artifact_sha256", {})[target_name] = _sha256_file(target_path)
    derived["derived_from"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "item_scope_sha256": _sha256_file(item_scope_path),
        "rule": "ADR-0021: keep targets on game items only",
    }
    derived_path = output_dir / "snapshot_game_targets.json"
    derived_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")

    filtered = {}
    for model_id, path in rankings.items():
        out = output_dir / f"{model_id}.jsonl"
        with path.open(encoding="utf-8") as source, out.open("w", encoding="utf-8") as sink:
            for line in source:
                if line.strip() and json.loads(line)["user_id"] in users:
                    sink.write(line)
        filtered[model_id] = str(out)
    summary = {
        "targets": targets.num_rows,
        "game_targets": kept.num_rows,
        "users_with_positive_game_target": len(users),
        "manifest": str(derived_path),
        "rankings": filtered,
    }
    (output_dir / "slice.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--item-scope", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], help="model_id=ranking.jsonl")
    args = parser.parse_args()
    rankings = dict((value.split("=", 1)[0], Path(value.split("=", 1)[1])) for value in args.model)
    print(
        json.dumps(
            build_slice(
                args.snapshot_manifest,
                item_scope_path=args.item_scope,
                output_dir=args.output_dir,
                rankings=rankings,
            )
        )
    )


if __name__ == "__main__":
    main()
