import random

import pytest

from trustrec.explanations.batch import build_claim_templates, explain_with_templates
from trustrec.explanations.claims import ClaimThresholds, select_explanation
from trustrec.explanations.evidence import aggregate_evidence

ASPECTS = ("story", "graphics", "value", "controls")
CUTOFF = 1_000


def _evidence(seed: int = 3) -> list[dict]:
    generator = random.Random(seed)
    rows = []
    for position in range(160):
        item = f"i{position % 4}"
        text = f"Review {position}: the {generator.choice(ASPECTS)} part is described here."
        start = text.index("the")
        rows.append(
            {
                "review_id": f"r{position}",
                "item_id": item,
                "author_id": f"u{generator.randint(0, 30)}",
                "timestamp": generator.randint(0, CUTOFF - 1),
                "aspect": generator.choice(ASPECTS),
                "sentiment": generator.choice(["positive", "negative", "neutral"]),
                "confidence": 1.0,
                "start": start,
                "end": len(text),
                "evidence": text[start:],
                "source_text": text,
            }
        )
    return rows


@pytest.mark.parametrize("max_claims", [1, 2])
def test_templates_match_select_explanation(max_claims: int) -> None:
    rows = _evidence()
    thresholds = ClaimThresholds(min_distinct_authors=3)
    priors = {"story": 0.1, "graphics": -0.05, "value": 0.0, "controls": 0.2}
    generator = random.Random(9)
    for item in ("i0", "i1", "i2", "i3"):
        item_rows = [row for row in rows if row["item_id"] == item]
        profiles = aggregate_evidence(
            item_rows, cutoff_timestamp=CUTOFF, prior_scores=priors, items=[item], aspects=ASPECTS
        )
        templates = build_claim_templates(item, profiles, item_rows, ASPECTS, CUTOFF, thresholds)
        for _ in range(5):
            weights = {aspect: generator.random() for aspect in ASPECTS}
            expected = select_explanation(
                item,
                profiles,
                item_rows,
                weights,
                CUTOFF,
                thresholds=thresholds,
                max_claims=max_claims,
            )
            actual = explain_with_templates(templates, weights, max_claims=max_claims)

            assert actual["status"] == expected.status
            assert [
                (c["aspect"], c["status"], c["sentiment_direction"]) for c in actual["claims"]
            ] == [(c.aspect, c.status, c.sentiment_direction) for c in expected.claims]
            for got, want in zip(actual["claims"], expected.claims, strict=True):
                assert got["score_contribution"] == pytest.approx(want.score_contribution)


def test_template_status_under_other_thresholds_and_cited_evidence() -> None:
    rows = _evidence()
    item_rows = [row for row in rows if row["item_id"] == "i0"]
    profiles = aggregate_evidence(item_rows, cutoff_timestamp=CUTOFF, items=["i0"], aspects=ASPECTS)
    strict = ClaimThresholds(min_distinct_authors=50)
    loose = ClaimThresholds(min_distinct_authors=1, min_support=0.0)
    templates = build_claim_templates("i0", profiles, item_rows, ASPECTS, CUTOFF, loose)

    for template in templates.values():
        assert template.status_under(loose) == template.status
        assert template.status_under(strict) == "insufficient_support"
        assert len(template.cited_evidence) <= 3
        for cited in template.cited_evidence:
            source = next(row for row in item_rows if row["review_id"] == cited["review_id"])
            assert source["source_text"][cited["start"] : cited["end"]] == cited["text"]
        if template.sentiment_direction in ("positive", "negative") and template.cited_evidence:
            assert template.cited_evidence[0]["sentiment"] == template.sentiment_direction
    cold = explain_with_templates(templates, {a: 0.25 for a in ASPECTS}, aspect_in_score=False)
    assert cold["aspect_in_score"] is False
    assert all(claim["score_contribution"] == 0.0 for claim in cold["claims"])


@pytest.mark.parametrize(
    ("positive", "negative", "direction"),
    [
        (9, 1, "positive"),
        (2, 1, "positive"),
        (1, 1, "conflicting"),
        (1, 2, "negative"),
        (0, 5, "negative"),
    ],
)
def test_claim_direction_follows_the_positive_share(positive, negative, direction) -> None:
    rows = []
    for index in range(positive + negative):
        sentiment = "positive" if index < positive else "negative"
        text = f"Review {index}: the story is {sentiment} here."
        rows.append(
            {
                "review_id": f"r{index}",
                "item_id": "i0",
                "author_id": f"u{index}",
                "timestamp": index,
                "aspect": "story",
                "sentiment": sentiment,
                "confidence": 1.0,
                "start": 0,
                "end": len(text),
                "evidence": text,
                "source_text": text,
            }
        )
    thresholds = ClaimThresholds(
        min_distinct_authors=1, min_support=0.0, min_effective_sample_size=1.0
    )
    profiles = aggregate_evidence(rows, cutoff_timestamp=CUTOFF, items=["i0"], aspects=["story"])
    claim = select_explanation(
        "i0", profiles, rows, {"story": 1.0}, CUTOFF, thresholds=thresholds
    ).claims[0]
    template = build_claim_templates("i0", profiles, rows, ["story"], CUTOFF, thresholds)["story"]

    assert claim.sentiment_direction == direction
    assert claim.conflicting == (direction == "conflicting")
    assert claim.positive_share == pytest.approx(positive / (positive + negative))
    assert claim.claim and "%" in claim.claim
    sides = [cited["sentiment"] for cited in template.cited_evidence]
    if positive and negative:
        assert set(sides) == {"positive", "negative"}
    lead = (
        "negative"
        if direction == "negative" or (direction == "conflicting" and negative > positive)
        else "positive"
    )
    assert sides[0] == lead
