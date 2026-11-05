"""Construct ranking baselines by model ID from one parameter set."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import METADATA_KEY, MatrixRanker
from .bpr_mf import BPRMFRanker
from .item_knn import ItemKNNRanker
from .matrix import InteractionMatrix
from .popularity import PopularityRanker, RecentPopularityRanker

BASELINE_MODEL_IDS = (
    "b0_most_popular",
    "b1_item_knn",
    "b2_bpr_mf",
    "b3_ppr",
    "b5_recent_popularity",
)
DEFAULT_BPR_SEED = 7


@dataclass(frozen=True)
class BaselineParameters:
    """Command-line parameters shared by the baseline ranking scripts."""

    positive_threshold: float = 4.0
    neighbor_limit: int = 0
    embedding_dimensions: int = 64
    learning_rate: float = 0.001
    regularization: float = 0.0001
    epochs: int = 50
    negative_samples: int = 1
    bpr_batch_size: int = 1024
    damping: float = 0.85
    max_iter: int = 100
    tol: float = 1e-12
    ppr_batch_size: int = 64
    window_days: float = 365.0
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def effective_seed(model_id: str, seed: int | None) -> int | None:
    """Return the seed a model uses; BPR MF defaults to a fixed seed."""

    if model_id == "b2_bpr_mf" and seed is None:
        return DEFAULT_BPR_SEED
    return seed


def create_baseline(model_id: str, parameters: BaselineParameters) -> MatrixRanker:
    """Create an unfitted baseline ranker."""

    if model_id not in BASELINE_MODEL_IDS:
        raise ValueError(f"unsupported model_id: {model_id}")
    if parameters.neighbor_limit < 0:
        raise ValueError("neighbor_limit cannot be negative")
    seed = effective_seed(model_id, parameters.seed)
    if model_id == "b0_most_popular":
        return PopularityRanker(positive_threshold=parameters.positive_threshold, seed=seed)
    if model_id == "b5_recent_popularity":
        return RecentPopularityRanker(
            window_days=parameters.window_days,
            positive_threshold=parameters.positive_threshold,
            seed=seed,
        )
    if model_id == "b1_item_knn":
        return ItemKNNRanker(
            positive_threshold=parameters.positive_threshold,
            n_neighbors=parameters.neighbor_limit or None,
            seed=seed,
        )
    if model_id == "b2_bpr_mf":
        return BPRMFRanker(
            embedding_dimensions=parameters.embedding_dimensions,
            learning_rate=parameters.learning_rate,
            regularization=parameters.regularization,
            positive_threshold=parameters.positive_threshold,
            epochs=parameters.epochs,
            negative_samples=parameters.negative_samples,
            seed=seed,
            batch_size=parameters.bpr_batch_size,
        )
    from trustrec.graph.personalized_pagerank import PersonalizedPageRankRanker

    return PersonalizedPageRankRanker(
        damping=parameters.damping,
        positive_threshold=parameters.positive_threshold,
        max_iter=parameters.max_iter,
        tol=parameters.tol,
        seed=seed,
        batch_size=parameters.ppr_batch_size,
    )


def ranker_class(model_id: str) -> type[MatrixRanker]:
    if model_id == "b0_most_popular":
        return PopularityRanker
    if model_id == "b1_item_knn":
        return ItemKNNRanker
    if model_id == "b5_recent_popularity":
        return RecentPopularityRanker
    if model_id == "b2_bpr_mf":
        return BPRMFRanker
    if model_id == "b3_ppr":
        from trustrec.graph.personalized_pagerank import PersonalizedPageRankRanker

        return PersonalizedPageRankRanker
    raise ValueError(f"unsupported model_id: {model_id}")


def read_model_file(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read model metadata and learned arrays without executing pickled code."""

    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if METADATA_KEY not in data.files:
            raise ValueError("model file has no metadata")
        metadata = json.loads(str(data[METADATA_KEY]))
        arrays = {name: data[name] for name in data.files if name != METADATA_KEY}
    return metadata, arrays


def load_baseline(path: Path, matrix: InteractionMatrix) -> MatrixRanker:
    """Rebuild a saved baseline on the matrix it was fitted on."""

    metadata, arrays = read_model_file(path)
    model_id = metadata.get("model_id")
    if model_id not in BASELINE_MODEL_IDS:
        raise ValueError(f"unsupported model_id in model file: {model_id}")
    ranker = ranker_class(model_id)(**metadata["init_parameters"])
    return ranker.restore(matrix, metadata, arrays)
