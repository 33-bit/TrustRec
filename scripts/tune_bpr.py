"""Tune BPR MF on the validation cutoff with a fixed, recorded trial budget."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.tune_baseline import tune_baseline
except ModuleNotFoundError:  # run as ``python scripts/tune_bpr.py``
    from tune_baseline import tune_baseline  # type: ignore[no-redef]


def tune_bpr(
    manifest_path: Path,
    *,
    output_path: Path,
    cutoff_name: str = "t0",
    learning_rates: Sequence[float] = (0.01, 0.05, 0.1),
    regularizations: Sequence[float] = (0.0001, 0.001, 0.01),
    epochs: Sequence[int] = (30,),
    embedding_dimensions: Sequence[int] = (64,),
    batch_size: int = 1024,
    positive_threshold: float = 4.0,
    seed: int = 7,
    workers: int = 1,
    bootstrap_samples: int = 200,
    save_best_path: Path | None = None,
) -> dict[str, Any]:
    """Run the BPR grid (dimensions x learning rate x regularization x epochs)."""

    grid = [
        {
            "embedding_dimensions": dimensions,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "epochs": epoch_count,
            "negative_samples": 1,
            "batch_size": batch_size,
        }
        for dimensions in embedding_dimensions
        for learning_rate in learning_rates
        for regularization in regularizations
        for epoch_count in epochs
    ]
    return tune_baseline(
        manifest_path,
        model_id="b2_bpr_mf",
        output_path=output_path,
        grid=grid,
        cutoff_name=cutoff_name,
        positive_threshold=positive_threshold,
        seed=seed,
        workers=workers,
        bootstrap_samples=bootstrap_samples,
        save_best_path=save_best_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument("--epochs", type=int, nargs="+", default=[30])
    parser.add_argument("--embedding-dimensions", type=int, nargs="+", default=[64])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--save-best", type=Path)
    args = parser.parse_args()
    result = tune_bpr(
        args.snapshot_manifest,
        output_path=args.output,
        learning_rates=args.learning_rate,
        regularizations=args.regularization,
        epochs=args.epochs,
        embedding_dimensions=args.embedding_dimensions,
        batch_size=args.batch_size,
        seed=args.seed,
        workers=args.workers,
        bootstrap_samples=args.bootstrap_samples,
        save_best_path=args.save_best,
    )
    print(json.dumps({"best": result["best"]["parameters"], "trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
