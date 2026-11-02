"""Sparse pre-cutoff interaction matrices shared by the ranking baselines."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

from .contracts import (
    CandidateSet,
    Timestamp,
    first_interactions_before_cutoff,
    timestamp_value,
)

MATRIX_SCHEMA_VERSION = "interaction-matrix-v1"
_SEPARATOR = "\x1f"


def rating_value(value: Any) -> float | None:
    """Return a finite rating, ``None`` for a missing rating, or raise."""

    if value is None or value == "":
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("rating must be numeric when provided")
    try:
        rating = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("rating must be numeric when provided") from error
    if not math.isfinite(rating):
        raise ValueError("rating must be finite")
    return rating


def _hash_strings(digest: Any, values: tuple[str, ...]) -> None:
    digest.update(str(len(values)).encode("ascii"))
    digest.update(_SEPARATOR.join(values).encode("utf-8"))


def catalog_fingerprint(item_ids: Iterable[str]) -> str:
    """Hash a pre-cutoff catalog independent of input order."""

    digest = hashlib.sha256(b"trustrec-catalog-v1")
    _hash_strings(digest, tuple(sorted(set(item_ids))))
    return digest.hexdigest()


def candidate_pool_signature(catalog_hash: str, history_item_ids: Iterable[str]) -> str:
    """Identify a ``catalog minus history`` candidate set without listing it."""

    digest = hashlib.sha256(b"trustrec-candidate-pool-v1")
    digest.update(catalog_hash.encode("ascii"))
    _hash_strings(digest, tuple(sorted(set(history_item_ids))))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class InteractionMatrix:
    """First pre-cutoff user-item events as sparse history and positive matrices.

    Users and items are indexed in ascending ID order, so index order is also
    the ``item_id_ascending`` tie-break order used by every baseline.
    """

    cutoff_timestamp: Timestamp
    positive_threshold: float
    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    history: sparse.csr_matrix
    positive: sparse.csr_matrix
    popularity: np.ndarray
    fingerprint: str
    catalog_hash: str
    user_index: dict[str, int] = field(repr=False)
    item_index: dict[str, int] = field(repr=False)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def _row(self, values: sparse.csr_matrix, user_id: str) -> np.ndarray:
        position = self.user_index.get(user_id)
        if position is None:
            return np.empty(0, dtype=np.int64)
        start, stop = values.indptr[position], values.indptr[position + 1]
        return values.indices[start:stop].astype(np.int64, copy=False)

    def history_indices(self, user_id: str) -> np.ndarray:
        """Return sorted item indices the user interacted with before the cutoff."""

        return self._row(self.history, user_id)

    def positive_indices(self, user_id: str) -> np.ndarray:
        """Return sorted item indices the user rated at or above the threshold."""

        return self._row(self.positive, user_id)

    def candidate_indices(self, user_id: str) -> np.ndarray:
        """Return catalog item indices minus the user's pre-cutoff history."""

        history = self.history_indices(user_id)
        mask = np.ones(self.n_items, dtype=bool)
        mask[history] = False
        return np.flatnonzero(mask)

    def candidate_signature(self, user_id: str) -> str:
        """Return the shared signature of this user's candidate pool."""

        history = self.history_indices(user_id)
        return candidate_pool_signature(
            self.catalog_hash, (self.item_ids[index] for index in history)
        )

    def candidate_set(self, user_id: str, snapshot_id: str) -> CandidateSet:
        """Build the protocol candidate set for one user from the fitted matrix."""

        return CandidateSet(
            user_id=user_id,
            snapshot_id=snapshot_id,
            cutoff_timestamp=self.cutoff_timestamp,
            candidate_item_ids=tuple(self.item_ids[i] for i in self.candidate_indices(user_id)),
            history_item_ids=frozenset(self.item_ids[i] for i in self.history_indices(user_id)),
        )


def _pairs_from_rows(
    interactions: Iterable[Mapping[str, Any]] | Any,
    cutoff_timestamp: Timestamp,
) -> tuple[list[str], list[str], list[float | None]]:
    rows = first_interactions_before_cutoff(interactions, cutoff_timestamp)
    return (
        [row["user_id"] for row in rows],
        [row["item_id"] for row in rows],
        [rating_value(row.get("rating")) for row in rows],
    )


def _require_id_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"interaction rows are missing {name}")
    column = frame[name]
    if column.isna().any():
        raise ValueError(f"interaction rows are missing {name}")
    if not column.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{name} must be a non-empty string")
    if (column.str.strip() == "").any():
        raise ValueError(f"{name} must be a non-empty string")
    return column


def _timestamp_column(frame: pd.DataFrame) -> np.ndarray:
    if "timestamp" not in frame.columns:
        raise ValueError("interaction rows are missing timestamp")
    column = frame["timestamp"]
    if column.isna().any():
        raise ValueError("interaction rows are missing timestamp")
    if pd.api.types.is_bool_dtype(column):
        raise ValueError("timestamp must be numeric or ISO formatted")
    if pd.api.types.is_datetime64_any_dtype(column):
        if getattr(column.dt, "tz", None) is None:
            raise ValueError("timestamp must be timezone-aware")
        nanoseconds = column.dt.tz_convert("UTC").astype("int64").to_numpy()
        return nanoseconds / 1_000_000.0
    if pd.api.types.is_numeric_dtype(column):
        values = column.to_numpy(dtype=np.float64)
    else:
        values = np.fromiter(
            (timestamp_value(value) for value in column), dtype=np.float64, count=len(column)
        )
    if not np.isfinite(values).all():
        raise ValueError("timestamp must be finite")
    return values


def _rating_column(frame: pd.DataFrame) -> np.ndarray:
    if "rating" not in frame.columns:
        return np.full(len(frame), np.nan)
    column = frame["rating"]
    if pd.api.types.is_bool_dtype(column):
        raise ValueError("rating must be numeric when provided")
    if pd.api.types.is_numeric_dtype(column):
        values = column.to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("rating must be finite")
        return values
    parsed = [rating_value(value) for value in column]
    return np.array([np.nan if value is None else value for value in parsed], dtype=np.float64)


def _canonical_row(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)


def _pairs_from_frame(
    frame: pd.DataFrame, cutoff_timestamp: Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = _require_id_column(frame, "user_id")
    items = _require_id_column(frame, "item_id")
    timestamps = _timestamp_column(frame)
    keep = timestamps < timestamp_value(cutoff_timestamp)
    review_ids = (
        frame["review_id"].astype(str).to_numpy()
        if "review_id" in frame.columns
        else np.full(len(frame), "", dtype=object)
    )
    work = pd.DataFrame(
        {
            "user_id": users.to_numpy()[keep],
            "item_id": items.to_numpy()[keep],
            "timestamp": timestamps[keep],
            "review_id": review_ids[keep],
            "position": np.flatnonzero(keep),
        }
    )
    tie_columns = ["user_id", "item_id", "timestamp", "review_id"]
    tied = work.duplicated(tie_columns, keep=False).to_numpy()
    canonical = np.full(len(work), "", dtype=object)
    if tied.any():
        original = frame.iloc[work["position"].to_numpy()[tied]]
        canonical[tied] = [_canonical_row(row) for row in original.to_dict("records")]
    work["canonical"] = canonical
    work = work.sort_values(
        ["timestamp", "review_id", "canonical"], kind="mergesort"
    ).drop_duplicates(["user_id", "item_id"], keep="first")
    ratings = _rating_column(frame.iloc[work["position"].to_numpy()])
    return work["user_id"].to_numpy(), work["item_id"].to_numpy(), ratings


def build_interaction_matrix(
    interactions: Iterable[Mapping[str, Any]] | pd.DataFrame | pa.Table,
    cutoff_timestamp: Timestamp,
    positive_threshold: float = 4.0,
) -> InteractionMatrix:
    """Keep the first event per pair before the cutoff and index it sparsely.

    Arrow tables and pandas frames use a vectorized path for full snapshots.
    Other iterables use the row contract in ``first_interactions_before_cutoff``.
    Both paths produce the same matrix and fingerprint.
    """

    if not math.isfinite(float(positive_threshold)):
        raise ValueError("positive_threshold must be finite")
    cutoff_value = timestamp_value(cutoff_timestamp)
    if isinstance(interactions, pa.Table):
        interactions = interactions.to_pandas()
    if isinstance(interactions, pd.DataFrame):
        user_values, item_values, ratings = _pairs_from_frame(interactions, cutoff_timestamp)
    else:
        user_list, item_list, rating_list = _pairs_from_rows(interactions, cutoff_timestamp)
        user_values = np.array(user_list, dtype=object)
        item_values = np.array(item_list, dtype=object)
        ratings = np.array(
            [np.nan if value is None else value for value in rating_list], dtype=np.float64
        )

    user_codes, user_uniques = pd.factorize(pd.Series(user_values, dtype=object), sort=True)
    item_codes, item_uniques = pd.factorize(pd.Series(item_values, dtype=object), sort=True)
    user_ids = tuple(str(value) for value in user_uniques)
    item_ids = tuple(str(value) for value in item_uniques)
    order = np.lexsort((item_codes, user_codes))
    user_codes = user_codes[order].astype(np.int64)
    item_codes = item_codes[order].astype(np.int64)
    ratings = ratings[order]
    with np.errstate(invalid="ignore"):
        positive_mask = ratings >= float(positive_threshold)
    shape = (len(user_ids), len(item_ids))
    history = sparse.csr_matrix(
        (np.ones(len(user_codes), dtype=np.float64), (user_codes, item_codes)), shape=shape
    )
    positive = sparse.csr_matrix(
        (
            np.ones(int(positive_mask.sum()), dtype=np.float64),
            (user_codes[positive_mask], item_codes[positive_mask]),
        ),
        shape=shape,
    )
    history.sort_indices()
    positive.sort_indices()
    popularity = np.bincount(item_codes[positive_mask], minlength=len(item_ids)).astype(np.int64)

    digest = hashlib.sha256(MATRIX_SCHEMA_VERSION.encode("ascii"))
    digest.update(repr((cutoff_value, float(positive_threshold))).encode("ascii"))
    _hash_strings(digest, user_ids)
    _hash_strings(digest, item_ids)
    digest.update(user_codes.tobytes())
    digest.update(item_codes.tobytes())
    digest.update(positive_mask.astype(np.uint8).tobytes())
    return InteractionMatrix(
        cutoff_timestamp=cutoff_timestamp,
        positive_threshold=float(positive_threshold),
        user_ids=user_ids,
        item_ids=item_ids,
        history=history,
        positive=positive,
        popularity=popularity,
        fingerprint=digest.hexdigest(),
        catalog_hash=catalog_fingerprint(item_ids),
        user_index={user_id: index for index, user_id in enumerate(user_ids)},
        item_index={item_id: index for index, item_id in enumerate(item_ids)},
    )


def top_k_order(
    primary: np.ndarray,
    tie_break: np.ndarray,
    k: int | None,
    secondary: np.ndarray | None = None,
) -> np.ndarray:
    """Return positions sorted by descending primary, descending secondary, then tie-break.

    With ``k`` it selects the top positions without sorting the whole array.
    """

    size = primary.size
    if k is None or k >= size:
        selected = np.arange(size)
    else:
        threshold = np.partition(primary, size - k)[size - k]
        selected = np.flatnonzero(primary >= threshold)
    keys = [tie_break[selected]]
    if secondary is not None:
        keys.append(-secondary[selected])
    keys.append(-primary[selected])
    ordered = selected[np.lexsort(keys)]
    return ordered if k is None else ordered[:k]
