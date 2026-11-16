"""Classify catalog items as game or non-game for explanations (ADR-0021)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

ITEM_SCOPE_VERSION = "adr0021-item-scope-v1"
STORE_CREDIT_TERMS = (
    "gift card",
    "gift code",
    "digital gift",
    "membership",
    "subscription",
    "robux",
    "v-bucks",
    "points card",
    "microsoft points",
    "nintendo points",
    "prepaid",
    "game time card",
    "psn card",
    "playstation network card",
    "live gold",
    "playstation plus",
    "switch online",
    "xbox game pass",
    "game pass ultimate",
    "ea access",
    "cash card",
    "eshop card",
    "wallet code",
    "steam wallet",
)
STORE_CREDIT_PATTERN = re.compile("|".join(re.escape(term) for term in STORE_CREDIT_TERMS), re.I)
MIN_LABELED_REVIEWS = 5
MIN_OUT_OF_SCOPE_SHARE = 0.5


def classify_item(title: str | None, labeled_reviews: int, out_of_scope_reviews: int) -> str:
    """Return ``game``, ``store_credit``, or ``hardware_accessory``."""

    if title and STORE_CREDIT_PATTERN.search(title):
        return "store_credit"
    if (
        labeled_reviews >= MIN_LABELED_REVIEWS
        and out_of_scope_reviews / labeled_reviews >= MIN_OUT_OF_SCOPE_SHARE
    ):
        return "hardware_accessory"
    return "game"


def classify_items(
    titles: Mapping[str, str | None],
    review_counts: Mapping[str, tuple[int, int]],
) -> dict[str, str]:
    """Classify every item in ``titles``.

    ``review_counts`` maps an item to (labeled pre-cutoff reviews, of which
    out of scope). Items without labeled reviews count as (0, 0).
    """

    return {
        item_id: classify_item(title, *review_counts.get(item_id, (0, 0)))
        for item_id, title in titles.items()
    }


def rules() -> dict[str, Any]:
    return {
        "version": ITEM_SCOPE_VERSION,
        "store_credit_terms": list(STORE_CREDIT_TERMS),
        "hardware_accessory": {
            "min_labeled_reviews": MIN_LABELED_REVIEWS,
            "min_out_of_scope_share": MIN_OUT_OF_SCOPE_SHARE,
        },
        "order": ["store_credit", "hardware_accessory"],
    }
