# T2.1 — Aspect Annotation Guideline

## Labeling unit

Annotate one sentence or clause at a time. Split a sentence when it contains two independent claims, for example a positive story claim and a negative performance claim. Preserve the original text and record character offsets for each evidence span.

## Aspect definitions

| Aspect | Include | Exclude |
| --- | --- | --- |
| `gameplay` | mechanics, pacing, engagement, difficulty | story quality or graphics alone |
| `story` | plot, characters, dialogue, narrative | gameplay mechanics |
| `graphics` | visuals, art style, animation, presentation | frame rate or crashes |
| `performance` | speed, bugs, crashes, lag, stability, hardware demands | subjective difficulty |
| `controls` | input, mapping, responsiveness, camera handling | general gameplay enjoyment |
| `multiplayer` | online/local play, matchmaking, player community | single-player content |
| `content_replay` | amount of content, length, endgame, replay value | monetary price |
| `value` | worth the money or time, price-to-experience judgment | absolute price without a value judgment |

Use `positive`, `negative`, or `neutral` sentiment for the aspect claim. A factual mention without an evaluative direction is `neutral`. Do not infer a user's personal interest from the polarity of one review. If an aspect is ambiguous, mark `needs_adjudication` rather than forcing a label.

## Adjudication protocol

Two annotators label at least 20% independently. They record aspect, sentiment, evidence span, and an out-of-scope flag. Disagreements are discussed against this guideline, and the adjudicator records the final label plus a short reason. Changes to this guideline apply only to train/development examples after the gold test is locked.

The pilot worksheet is intentionally unadjudicated until the team supplies two independent label columns. Weak keyword suggestions are convenience hints only and must not be used as gold labels.

