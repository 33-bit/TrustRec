# Task: T7.2: Evidence-removal faithfulness of real explanations

Status: done for the ADR-0017 H0 top-10 on `t1`. The coverage numbers it
depends on stay provisional until the T7.3 audit (ADR-0010).

## Method

The test is part of `scripts/run_explanations_batch.py` (same command as
[T7.1](t7_1-explanations.md)). For each recommendation with a supported claim
whose aspect is in the ranking score:

1. **Cited removal.** Remove every review that mentions the claim aspect for
   the item, then recompute the item's aspect profile.
2. **Random removal.** Remove the same number of the item's other evidence
   reviews, drawn with a seed fixed per (item, aspect). When the item has
   fewer other reviews, remove all of them and mark the pool as short.
3. **Score change.** Recompute the user's aspect score
   `A(u, i) = sum_a w_ua * deviation_ia` and the H0 aspect part, keeping the
   user's candidate minimum and maximum fixed.
4. **Top-10 check.** Compare the new total with the score of the 11th item.

A claim is `faithful` when the cited removal changes the score more than the
random removal. The T4.2 `audit_faithfulness` draws its random set from the
claim's own evidence. When a claim cites all of that evidence, the two removals
are the same set, so that audit cannot be used here.

Summary SHA-256:
`076607e3b1b7c5a81c6c4b7f1866f3da168f1ab0871c8821d8a40104d91833d0`.
Recommendations file SHA-256:
`68ee2f606b078204338df91180b0b92ad03c9703fc2ba16c71b25f74aab58d7f`.
The explanations equal the second T7.1 run; this run only adds the
`faithfulness` field.

## Results

| Group | Recommendations |
| --- | --- |
| Not applicable: aspect not in the score (cold-start fallback) | 541,208 (87.6%) |
| No supported claim | 2,178 |
| Tested | 74,364 |

| Tested claims | Tested | Faithful | Score moves in the claimed direction | Leaves the top-10 after cited removal | After random removal |
| --- | --- | --- | --- | --- | --- |
| All | 74,364 | 98.3% | — | 14.1% | 0.9% |
| Positive | 62,096 | 98.0% | 98.8% | 15.9% | 0.6% |
| Negative | 1,587 | 98.9% | 98.7% | 0.3% | 6.7% |
| Mixed | 10,681 | 99.8% | not defined | 5.2% | 1.8% |
| Random pool full size | 46,563 | 99.2% | — | 8.6% | 0.8% |
| Random pool short | 27,801 | 96.9% | — | 23.3% | 1.2% |

The median absolute change in the aspect part is 0.0193 for cited removal and
0.0015 for random removal.

## Findings

- When the aspect is part of the score, the cited aspect evidence drives it:
  removing it changes the score about 13 times as much as random removal, and
  moves the score in the claimed direction 98.8% of the time.
- For 14.1% of tested recommendations, the item would leave the top-10
  without the cited evidence; for the rest, recency and graph keep it there.
  This matches ADR-0010: the explanation must say when aspect is only one score
  part, and `dominant_score_part` records this.
- For a negative claim, removing the negative evidence raises the score, so
  the item never leaves the top-10 by that removal (0.3%).
- The result holds when the random pool is full size (99.2% faithful). A short
  pool removes fewer reviews, which makes the test easier, so those 27,801
  claims are reported apart.
- For 87.6% of recommendations (users without positive history), the claim
  does not explain the ranking at all, because the ranking is B5. Those claims
  are descriptive only. Reports must say so.
