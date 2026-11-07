# Task: T7.1: Explanations for the ADR-0017 H0 top-10 on `t1`

Status: second run done under [ADR-0019](../adr/0019-claim-direction-by-evidence-share.md).
The numbers stay provisional, because ADR-0010 forbids reporting coverage
without the T7.3 correctness audit. The first run is kept below as a record.

## Context

- Decision: [ADR-0018](../adr/0018-explanation-first-research-focus.md)
- Specs: [evidence and explanation](../specs/evidence-and-explanation.md), [result reporting](../specs/result-reporting.md)
- Code: `src/trustrec/explanations/batch.py`, `scripts/run_explanations_batch.py`

```bash
PYTHONPATH=src python3 scripts/run_explanations_batch.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --evidence artifacts/evidence/t1_gpt-6-luna.jsonl --ranking artifacts/rankings/h0_t1_adr0017.jsonl --h0 '{"mf_weight": 0.0, "graph_weight": 0.2, "recency_weight": 0.6, "aspect_weight": 0.2}' --mf-model artifacts/models/b2_tuned_t1_seed07.npz --graph-params '{"damping": 0.85, "max_iter": 100}' --normalization minmax --cold-start fallback --cold-window-days 60 --recency-window-days 60 --k 10 --max-claims 1 --workers 4 --output-dir artifacts/explanations/t1_h0_adr0017_k10
```

Settings: default `ClaimThresholds` (support 0.25, 5 distinct authors,
effective sample size 2, confidence 0.5), one claim per recommendation,
shrinkage 5, curated gpt-6-luna evidence. The run took 12 minutes on 4
cores. Summary SHA-256: `4b6da623b97debaaab9ada6734c3597a8d5051e409cfd1fda995e43f4193285f`. Recommendations file SHA-256: `f5e4e20cc49f03a93d1a56523b86d3fc591488cf6425f725aeeb987d5d7fefd7`.

## Checks

- The recomputed top-10 equals the saved ranking file for all 61,775 users,
  so the explanations do not change the ranking.
- Explanation profiles equal the ranker's aspect profiles (largest gap 3e-17).
- Every cited passage was checked against the full review text.

## Results (provisional)

617,750 recommendations for 1,729 distinct items.

| Group | Recommendations | `supported` |
| --- | --- | --- |
| All | 617,750 | 99.6% |
| Aspect in the score (users with history) | 76,220 | 97.6% |
| Cold-start fallback (aspect not in the score) | 541,530 | 99.9% |
| Items with 1–9 positive interactions | 156 | 16.7% |
| Items with 10 or more | 617,594 | about 99.7% |

- Every user has at least one supported item in the top-10.
- No recommendation is `unavailable`.
- The most common refusal reason is "fewer than 5 distinct authors" (2,178).
- For users with history, the largest score part is recency for 51,035
  recommendations, aspect for 15,983, and graph for 9,202.
- For 88% of recommendations (cold-start users), the aspect term plays no part
  in the ranking. Their claims describe the item, not why it was recommended.
- Coverage barely moves under other thresholds: 99.3% to 99.9% across
  3–10 distinct authors and support 0.1–0.5.

## Findings

- **Almost every supported claim is "mixed".** 615,042 of 615,572 supported
  claims have direction `conflicting`, 529 are positive, and 1 is negative.
  The current rule marks a claim as conflicting when any review on either
  side exists (`positive_mass > 0` and `negative_mass > 0`). Popular items
  always have some opposing reviews: the median selected (item, aspect) pair
  has 91 evidence rows.
- Weighted by recommendations, the positive share of polar evidence behind
  the selected claims is:
  - below 1/3 for 0.3%;
  - 1/3 to 2/3 for 19.9%;
  - 2/3 to 0.9 for 41.1%;
  - at least 0.9 for 38.7%.

  About 80% of the "mixed" claims are clearly positive.
- High coverage is therefore not a useful result yet. Coverage is high
  because recommended items are popular and have many reviews, and the claims
  say little. A direction rule based on the share of polar evidence is needed
  before T7.2 and T7.3. That rule changes claim semantics and needs an ADR.

## Second run under ADR-0019 (2026-11-07)

Claim direction now comes from the positive share of polar evidence. The run
used the same command, settings, and checks. All checks pass. Summary SHA-256:
`95055c49eaf983c6f99bc5714e1a736fce7b2ae044e401204e51ac083a0f06a7`.
Recommendations file SHA-256:
`abc9de12ccaa91f1b5ab789fef01a0898024062d1a0280dd5951d06daed51ddc`. The first
run is in `artifacts/explanations/t1_h0_adr0017_k10_v1_any_conflict`.

Coverage does not change (99.6% `supported`), because the direction rule does
not change support.

| Direction of supported claims | Aspect in the score | Cold-start fallback | All |
| --- | --- | --- | --- |
| Positive | 62,096 | 485,963 | 548,059 (89.0%) |
| Mixed | 10,681 | 54,868 | 65,549 (10.6%) |
| Negative | 1,587 | 377 | 1,964 (0.3%) |

Two more findings:

- **Gameplay dominates.** 596,262 of 615,572 supported claims (96.9%) are
  about gameplay. The rest are performance 6,882, value 4,572, content and
  replay 3,052, controls 1,645, story 1,421, graphics 957, and multiplayer 781.
  Claim selection ranks aspects by the user weight times the item deviation.
  Gameplay has the largest prior weight (it is the most mentioned aspect), so
  it wins for almost every user. T7.3 should check whether this makes the
  explanations repetitive.
- Mixed claims now cite both sides. For example, `B000006RGR` graphics (41%
  positive) cites "you've probably experienced better game graphics", "The
  background is not so good", and "how good the game layouts and graphics
  were."
