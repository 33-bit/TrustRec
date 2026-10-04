# T2.1: aspect label guideline

The LLM labels one sentence or clause at a time. Split a sentence when it has two separate claims. Keep the original text and record character offsets for each evidence span.

## Aspect definitions

| Aspect | Include | Exclude |
| --- | --- | --- |
| `gameplay` | Mechanics, pacing, engagement, and difficulty. | Story or graphics alone. |
| `story` | Plot, characters, dialogue, and narrative. | Gameplay mechanics. |
| `graphics` | Visuals, art style, animation, and presentation. | Frame rate and crashes. |
| `performance` | Speed, bugs, crashes, lag, stability, and hardware demands. | Subjective difficulty. |
| `controls` | Input, mapping, responsiveness, and camera handling. | General enjoyment. |
| `multiplayer` | Online or local play, matchmaking, and the game community. | Single-player content. |
| `content_replay` | Content amount, length, endgame, and replay value. | Price alone. |
| `value` | Worth of the money or time. | A price without a value claim. |

Use `positive`, `negative`, or `neutral` for the claim. Use `out_of_scope` for accessories, shipping, packaging, sellers, and unrelated hardware. Do not infer an aspect from a star rating. Let the LLM abstain when the text is unclear.

## Pilot and pseudo-test rules

The 100-unit pilot has split role `development_pilot` and status `llm_silver`. Use it to refine the prompt, guideline, threshold, or model settings. Do not use it for recommendation test scoring.

Create an independent `llm_pseudo_test` pass after the pilot. Freeze its source units and provenance before tuning. Use it only for aspect, sentiment, evidence, and explanation consistency checks.
