"""LLM-judge audit of explanation claims (T7.3).

The judge sees the claim, the aspect definition, and the cited passages, but
not the labeler's sentiment labels. It reports, for each passage, whether it
discusses the aspect, its polarity, and whether it refers to the reviewed
product; and for the claim, whether the passages support it. Results are LLM
agreement, not human judgment or ground truth (explanation audit contract).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

JUDGE_PROMPT_VERSION = "t7.3-claim-judge-v1"
ASPECT_DEFINITIONS = {
    "gameplay": "mechanics, pacing, difficulty, and play experience",
    "story": "plot, characters, dialogue, campaign, and narrative",
    "graphics": "visuals, art, animation, and presentation",
    "performance": "bugs, crashes, freezes, lag, loading, and stability",
    "controls": "input, mapping, responsiveness, camera, and handling",
    "multiplayer": "online play, local play, matchmaking, and player interaction",
    "content_replay": "content amount, length, endgame, and replay value",
    "value": "price, cost, worth, and value for money or time",
}
POLARITIES = ("positive", "negative", "neutral")
SUPPORT = ("yes", "partly", "no")
JUDGE_PROMPT = f"""You audit explanations for video game recommendations.

You receive one claim about one aspect of one product, the aspect definition, and up to three
review passages cited as evidence. Judge only from the passages.

For each passage decide:
- `about_aspect`: does the passage discuss the named aspect as defined?
- `polarity`: the passage's opinion on that aspect: one of {", ".join(POLARITIES)}.
- `about_product`: does the passage describe the reviewed product, not another product?

Then decide `claim_supported`: do the passages, read together, support the claim's direction?
Use `yes`, `partly` (some passages support it, others are off-aspect or opposite), or `no`.
A positive or negative claim may cite one opposing passage on purpose; that alone does not
make it `no`. A mixed claim is supported (`yes`) when the passages show both positive and
negative opinions on the aspect.

Return one JSON object:
{{"passages": [{{"id": <id>, "about_aspect": <bool>, "polarity": <polarity>,
"about_product": <bool>}}], "claim_supported": <yes|partly|no>}}"""


def judge_prompt_hash() -> str:
    return hashlib.sha256(f"{JUDGE_PROMPT_VERSION}\n{JUDGE_PROMPT}".encode()).hexdigest()


def sample_claims(
    recommendation_rows: Iterable[Mapping[str, Any]],
    templates: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    per_stratum: int,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Sample supported (item, aspect) claims, stratified by direction.

    Each sampled claim keeps how many recommendations used it, so results can
    be read per claim or weighted by recommendations.
    """

    uses: dict[tuple[str, str], int] = {}
    for row in recommendation_rows:
        for claim in row.get("claims", []):
            if claim["status"] == "supported":
                key = (row["item_id"], claim["aspect"])
                uses[key] = uses.get(key, 0) + 1
    strata: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(uses):
        strata.setdefault(templates[key]["sentiment_direction"], []).append(key)
    generator = random.Random(seed)
    sample = []
    for direction in sorted(strata):
        keys = strata[direction]
        chosen = sorted(generator.sample(keys, min(per_stratum, len(keys))))
        for key in chosen:
            sample.append(
                {
                    **templates[key],
                    "stratum": direction,
                    "stratum_size": len(keys),
                    "recommendation_uses": uses[key],
                }
            )
    return sample


def claim_statement(claim: Mapping[str, Any]) -> str:
    share = claim.get("positive_share")
    direction = claim["sentiment_direction"]
    aspect = claim["aspect"]
    if share is None:
        return f"The product has {direction} evidence for {aspect}."
    if direction == "conflicting":
        return f"Opinions on {aspect} are mixed: {share:.0%} positive."
    if direction == "negative":
        return f"Opinions on {aspect} are mostly negative: {1 - share:.0%} negative."
    return f"Opinions on {aspect} are mostly positive: {share:.0%} positive."


def judge_input(claim: Mapping[str, Any]) -> str:
    """Return the user message for one claim. Labeler polarities are withheld."""

    return json.dumps(
        {
            "aspect": claim["aspect"],
            "aspect_definition": ASPECT_DEFINITIONS[claim["aspect"]],
            "claim": claim_statement(claim),
            "passages": [
                {"id": position, "text": passage["text"]}
                for position, passage in enumerate(claim["cited_evidence"])
            ],
        },
        ensure_ascii=False,
    )


def parse_judgment(content: str, passage_count: int) -> dict[str, Any]:
    """Validate a judge response; raise ValueError when it is incomplete."""

    from trustrec.nlp.llm_labeler import extract_json_object

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = extract_json_object(content)
    if not isinstance(payload, dict):
        raise ValueError("judgment is not a JSON object")
    support = payload.get("claim_supported")
    if support not in SUPPORT:
        raise ValueError(f"invalid claim_supported: {support}")
    by_id = {
        entry["id"]: entry
        for entry in payload.get("passages", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    }
    passages = []
    for position in range(passage_count):
        entry = by_id.get(position)
        if entry is None:
            raise ValueError(f"judgment lacks passage {position}")
        if entry.get("polarity") not in POLARITIES:
            raise ValueError(f"invalid polarity for passage {position}")
        passages.append(
            {
                "about_aspect": bool(entry.get("about_aspect")),
                "polarity": entry["polarity"],
                "about_product": bool(entry.get("about_product")),
            }
        )
    return {"passages": passages, "claim_supported": support}


def summarize_judgments(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate audit results per stratum and overall (LLM agreement)."""

    def block(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        judged = [row for row in items if row.get("judgment")]
        passages = [
            (cited, verdict)
            for row in judged
            for cited, verdict in zip(
                row["cited_evidence"], row["judgment"]["passages"], strict=True
            )
        ]
        support = [row["judgment"]["claim_supported"] for row in judged]
        weight = sum(row["recommendation_uses"] for row in judged)
        weighted_yes = sum(
            row["recommendation_uses"]
            for row in judged
            if row["judgment"]["claim_supported"] == "yes"
        )
        return {
            "claims": len(items),
            "judged": len(judged),
            "failed": len(items) - len(judged),
            "passages": len(passages),
            "aspect_correct": _share(verdict["about_aspect"] for _, verdict in passages),
            "about_product": _share(verdict["about_product"] for _, verdict in passages),
            "polarity_agrees_with_label": _share(
                verdict["polarity"] == cited["sentiment"] for cited, verdict in passages
            ),
            "claim_supported": {value: _share(s == value for s in support) for value in SUPPORT},
            "claim_supported_yes_weighted_by_recommendations": weighted_yes / weight
            if weight
            else None,
        }

    strata = sorted({row["stratum"] for row in rows})
    return {
        "all": block(rows),
        **{f"stratum:{name}": block([r for r in rows if r["stratum"] == name]) for name in strata},
    }


def _share(values: Iterable[bool]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
