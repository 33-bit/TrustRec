"""LLM verification of display passages (ADR-0020).

Each displayed claim keeps candidate passages from both sides. A verifier
model rates every candidate for aspect, polarity, and product, without seeing
the labeler's polarity. Only verified passages are shown. A claim with no
verified passage on its side abstains with ``no_verified_passage``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .batch import DISPLAY_EVIDENCE, claim_side
from .judge import ASPECT_DEFINITIONS, POLARITIES

VERIFY_PROMPT_VERSION = "adr0020-passage-verify-v1"
VERIFY_PROMPT = f"""You check review passages before they are shown as evidence for a video game.

You receive one aspect, its definition, and numbered passages from reviews of one product.
For each passage decide:
- `about_aspect`: does the passage discuss this aspect as defined? Generic praise or
  complaints about the whole product ("love it", "great game", "hate it") are not about the
  aspect unless they name something the definition covers.
- `polarity`: the passage's opinion on the aspect: one of {", ".join(POLARITIES)}.
- `about_product`: does the passage describe the reviewed product, not another product?

Return one JSON object:
{{"passages": [{{"id": <id>, "about_aspect": <bool>, "polarity": <polarity>,
"about_product": <bool>}}]}}"""


def verify_prompt_hash() -> str:
    return hashlib.sha256(f"{VERIFY_PROMPT_VERSION}\n{VERIFY_PROMPT}".encode()).hexdigest()


def passage_key(aspect: str, passage: Mapping[str, Any]) -> str:
    return f"{passage['review_id']}|{aspect}|{passage['start']}|{passage['end']}"


def verify_input(aspect: str, passages: Sequence[Mapping[str, Any]]) -> str:
    """Return the user message for one aspect. Labeler polarities are withheld."""

    return json.dumps(
        {
            "aspect": aspect,
            "aspect_definition": ASPECT_DEFINITIONS[aspect],
            "passages": [
                {"id": position, "text": passage["text"]}
                for position, passage in enumerate(passages)
            ],
        },
        ensure_ascii=False,
    )


def parse_verdicts(content: str, count: int) -> list[dict[str, Any]]:
    """Validate a verifier response; raise ValueError when it is incomplete."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("verdicts are not JSON") from error
    entries = payload.get("passages") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("verdicts have no passages list")
    by_id = {
        entry["id"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    }
    verdicts = []
    for position in range(count):
        entry = by_id.get(position)
        if entry is None or entry.get("polarity") not in POLARITIES:
            raise ValueError(f"missing or invalid verdict for passage {position}")
        verdicts.append(
            {
                "about_aspect": bool(entry.get("about_aspect")),
                "polarity": entry["polarity"],
                "about_product": bool(entry.get("about_product")),
            }
        )
    return verdicts


def select_verified(
    claim: Mapping[str, Any], verdicts: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    """Return the passages to show and whether the claim keeps its support.

    ``verdicts`` maps ``passage_key`` to a verdict for every candidate.
    """

    aspect = claim["aspect"]
    candidates = claim.get("candidate_evidence") or []
    positive = sum(1 for passage in candidates if passage["sentiment"] == "positive")
    negative = sum(1 for passage in candidates if passage["sentiment"] == "negative")
    side = claim_side(claim["sentiment_direction"], positive, negative)
    if side is None:
        return [], False
    opposite = "negative" if side == "positive" else "positive"

    def passing(polarity: str) -> list[dict[str, Any]]:
        chosen = []
        for passage in candidates:
            verdict = verdicts[passage_key(aspect, passage)]
            if (
                verdict["about_aspect"]
                and verdict["about_product"]
                and verdict["polarity"] == polarity
            ):
                chosen.append({**passage, "verified_polarity": verdict["polarity"]})
        return chosen

    main = passing(side)[:2]
    other = passing(opposite)[:1]
    shown = (main + other)[:DISPLAY_EVIDENCE]
    if claim["sentiment_direction"] == "conflicting":
        return shown, bool(main) and bool(other)
    return shown, bool(main)
