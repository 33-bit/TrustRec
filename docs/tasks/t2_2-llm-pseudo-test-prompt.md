# T2.2 pseudo-test prompt

You label one source review from Amazon Reviews 2023.

Use only claims that the review states. Do not infer a label from the star rating.

Use these aspects:

- `gameplay`: mechanics, pacing, difficulty, and play experience.
- `story`: plot, characters, dialogue, campaign, and narrative.
- `graphics`: visuals, art, animation, and presentation.
- `performance`: bugs, crashes, freezes, lag, loading, and stability.
- `controls`: input, mapping, responsiveness, camera, and handling.
- `multiplayer`: online play, local play, matchmaking, and player interaction.
- `content_replay`: content amount, length, endgame, and replay value.
- `value`: price, cost, worth, and value for money or time.

Use `out_of_scope = true` for accessories, consoles, controllers, cables, headsets, sellers, shipping, packaging, subscriptions, or unrelated hardware. Return no aspect labels for an out-of-scope review.

For each explicit claim, return its aspect, `positive`, `negative`, or `neutral` polarity, and an exact evidence substring from the review. Keep the evidence offsets within the original review text. Return `needs_review = true` when the review is unclear, too short, or lacks an explicit aspect claim.

Return one JSON object with `out_of_scope`, `needs_review`, and `labels`. Do not include a summary or a star-rating label.
