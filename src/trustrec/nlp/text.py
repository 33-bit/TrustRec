"""Text segmentation helpers that preserve offsets in the source review."""

from __future__ import annotations

import re
from dataclasses import dataclass

_BOUNDARY = re.compile(
    r"<br\s*/?>|(?<=[.!?])\s+|,\s+(?:but|however|although)\s+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TextUnit:
    """A clause with character offsets into the original review text."""

    unit_id: str
    review_id: str
    text: str
    start: int
    end: int


def _trimmed_span(source: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace and a removed contrast comma from a source span."""

    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    while end > start and source[end - 1] in ",;":
        end -= 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def split_text_units(
    text: str,
    *,
    review_id: str = "",
    unit_id_prefix: str = "unit",
) -> tuple[TextUnit, ...]:
    """Split a review into clauses and retain exact source offsets.

    HTML line breaks, sentence endings, and common contrast conjunctions create
    boundaries. The returned unit text is always a direct slice of ``text``.
    """

    if not isinstance(text, str) or not text.strip():
        return ()

    units: list[TextUnit] = []
    segment_start = 0
    for match in _BOUNDARY.finditer(text):
        span_start, span_end = _trimmed_span(text, segment_start, match.start())
        if span_start < span_end:
            units.append(
                TextUnit(
                    unit_id=f"{unit_id_prefix}-{len(units) + 1:03d}",
                    review_id=review_id,
                    text=text[span_start:span_end],
                    start=span_start,
                    end=span_end,
                )
            )
        segment_start = match.end()

    span_start, span_end = _trimmed_span(text, segment_start, len(text))
    if span_start < span_end:
        units.append(
            TextUnit(
                unit_id=f"{unit_id_prefix}-{len(units) + 1:03d}",
                review_id=review_id,
                text=text[span_start:span_end],
                start=span_start,
                end=span_end,
            )
        )
    return tuple(units)
