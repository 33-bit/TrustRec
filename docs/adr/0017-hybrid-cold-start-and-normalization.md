# ADR-0017: Hybrid cold-start fallback and min-max normalization

Status: Proposed. Date: 2026-11-05. Amends [ADR-0009](0009-score-normalization-and-gating.md) if accepted.

## Context

ADR-0009 normalizes MF, graph, and aspect scores as percentile ranks within
each user's candidate set. On the full Video Games snapshot this fails in two
ways (see T6.4):

1. For users without positive history (88% of test users), MF and graph both
   fall back to popularity, and the user's aspect weights equal the global
   prior. The hybrid then gives every such user the same non-personalized list,
   which is a mix of popularity and aspect quality.
2. Percentiles erase the gaps in heavy-tailed scores. Near the top of the
   popularity list, percentile scores differ by about 1e-5, so even a small
   aspect term (T0 mean gate 0.056) reorders the top 10.

Every H0 and T0 setting was below popularity on `t0` validation. Both beat B0
on `t1` test, and the gain came only from users without history.

## Proposed decision

Add two scorer settings to `HybridBatchScorer` and `scripts/run_hybrid_batch.py`.
Both settings keep their ADR-0009 values by default.

- `--cold-start fallback`: users without positive history get the baseline
  order with no aspect term and a zero gate. The baseline is B5 with
  `--cold-window-days` set to B5's tuned window, or B0 if no window is given.
  Users with history keep the hybrid formula.
- `--normalization minmax`: scale each component by the candidate minimum and
  maximum instead of percentiles. All-equal values map to 0.5.

Tune H0 and T0 on `t0` again with the same grids (9 trials each) under the
proposed settings. Report both the ADR-0009 runs and the new runs. Do not
select between them on `t1`.

## Consequences

- With the fallback, H0 and T0 equal B5 for users without history. Any
  difference from B5 then comes from users with history, where aspect evidence
  can personalize.
- ADR-0009 results stay in the record and are not overwritten.
- If the project owners reject this ADR, the settings stay unused and the
  ADR-0009 results are the reported hybrid results.
