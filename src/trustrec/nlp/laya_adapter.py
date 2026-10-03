"""Laya adapter for structured, local-first review annotation.

Laya is optional. Importing this module does not import or download a model;
the runner imports ``laya.Router`` only when inference is requested.
"""

from __future__ import annotations

from typing import Any

ASPECTS = (
    "gameplay",
    "story",
    "graphics",
    "performance",
    "controls",
    "multiplayer",
    "content_replay",
    "value",
)
SENTIMENTS = ("positive", "negative", "neutral")
ASPECT_DEFINITIONS = {
    "gameplay": "game mechanics, engagement, pacing or difficulty",
    "story": "plot, characters, dialogue or narrative",
    "graphics": "visual quality, art style or animation (not frame rate)",
    "performance": "game bugs, crashes, lag, frame rate or stability",
    "controls": "in-game input, responsiveness or camera handling",
    "multiplayer": "online or local multiplayer, co-op or matchmaking",
    "content_replay": "amount of content, length, endgame or replay value",
    "value": "whether a game is worth its money or time (not price alone)",
}


def build_questions() -> dict[str, dict[str, Any]]:
    """Build the fixed typed decision schema used for every unit."""

    questions: dict[str, dict[str, Any]] = {}
    for aspect in ASPECTS:
        display_aspect = aspect.replace("_", " ")
        questions[f"has_{aspect}"] = {
            "type": "noul",
            "instructions": (
                f"Does this text explicitly discuss {ASPECT_DEFINITIONS[aspect]} "
                "of a software game? Include factual mentions. Exclude accessory hardware "
                "and opinions only about a different product or game."
            ),
        }
        questions[f"sentiment_{aspect}"] = {
            "type": "choice",
            "instructions": f"What sentiment does this text express about {display_aspect}?",
            "criteria": {
                "positive": "praises or expresses satisfaction with this aspect",
                "negative": "criticizes or expresses dissatisfaction with this aspect",
                "neutral": "mentions this aspect without positive or negative evaluation",
            },
        }
    questions["out_of_scope"] = {
        "type": "noul",
        "instructions": (
            "Is this text about an accessory, shipping, packaging, seller, "
            "or another non-game product matter?"
        ),
    }
    questions["needs_adjudication"] = {
        "type": "noul",
        "instructions": (
            "Is the intended product aspect or sentiment ambiguous enough "
            "that a human should review it?"
        ),
    }
    return questions


def _answer_confidence(answer: dict[str, Any]) -> float | None:
    # Laya's entropy confidence is not the selected answer's probability.
    value = answer.get("answer_confidence")
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _answer_value(answer: dict[str, Any], key: str) -> Any:
    value = answer.get(key)
    if isinstance(value, dict):
        return value.get("value", value.get("probability"))
    return value


def normalize_prediction(
    text: str,
    prediction: dict[str, Any],
    *,
    min_presence: float = 0.85,
    min_confidence: float = 0.70,
) -> dict[str, Any]:
    """Convert a Laya response into TrustRec's preliminary label shape."""

    del text  # retained in the runner for source hashes and evidence spans
    answers = prediction.get("answers", {})
    labels = []
    presence_probability: dict[str, float] = {}
    confidence: dict[str, float] = {}
    review_reasons: list[str] = []

    def check_answer(name: str, answer: dict[str, Any]) -> None:
        if not answer:
            review_reasons.append(f"missing:{name}")
        value = _answer_confidence(answer)
        if value is None:
            review_reasons.append(f"missing_confidence:{name}")
        elif value < min_confidence or answer.get("low_confidence"):
            review_reasons.append(f"low_confidence:{name}")

    for aspect in ASPECTS:
        presence_answer = answers.get(f"has_{aspect}", {})
        check_answer(f"has_{aspect}", presence_answer)
        probability = _answer_value(presence_answer, "noul")
        if isinstance(probability, (float, int)):
            presence_probability[aspect] = float(probability)
        presence_confidence = _answer_confidence(presence_answer)
        if presence_confidence is not None:
            confidence[f"has_{aspect}"] = presence_confidence
        if not isinstance(probability, (float, int)) or float(probability) < min_presence:
            continue
        sentiment_answer = answers.get(f"sentiment_{aspect}", {})
        check_answer(f"sentiment_{aspect}", sentiment_answer)
        polarity = _answer_value(sentiment_answer, "choice")
        if polarity not in SENTIMENTS:
            raise ValueError(f"Unknown sentiment for {aspect}: {polarity!r}")
        sentiment_confidence = _answer_confidence(sentiment_answer)
        if sentiment_confidence is not None:
            confidence[f"sentiment_{aspect}"] = sentiment_confidence
        labels.append({"aspect": aspect, "polarity": polarity})

    out_of_scope_answer = answers.get("out_of_scope", {})
    adjudication_answer = answers.get("needs_adjudication", {})

    def positive_decision(answer: dict[str, Any]) -> bool:
        value = _answer_value(answer, "noul")
        if isinstance(value, bool):
            return value
        return isinstance(value, (float, int)) and float(value) >= 0.70

    out_of_scope = positive_decision(out_of_scope_answer)
    needs_adjudication = positive_decision(adjudication_answer)
    for name, answer in (
        ("out_of_scope", out_of_scope_answer),
        ("needs_adjudication", adjudication_answer),
    ):
        check_answer(name, answer)
        answer_confidence = _answer_confidence(answer)
        if answer_confidence is not None:
            confidence[name] = answer_confidence
    routing = prediction.get("routing", {})
    return {
        "labels": labels,
        "presence_probability": presence_probability,
        "confidence": confidence,
        "out_of_scope": out_of_scope,
        "needs_adjudication": needs_adjudication or bool(review_reasons),
        "review_reasons": review_reasons,
        "model": routing.get("model"),
        "raw_prediction": prediction,
    }


def create_router(
    *, model: str = "laya", device: str | None = None, revision: str | None = None
) -> Any:
    """Create a Laya router lazily with a useful missing-dependency message."""

    try:
        from laya import Router
    except ImportError as error:
        raise RuntimeError(
            "Laya is optional. Install it with `python -m pip install -e '.[laya]'` "
            "before running the local labeling command."
        ) from error
    kwargs = {"device": device} if device else {}
    if revision:
        kwargs["revision"] = revision
    router = Router(**kwargs)
    return router, model
