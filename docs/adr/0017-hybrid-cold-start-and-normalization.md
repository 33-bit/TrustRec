# ADR-0017: Hybrid recency term, cold-start fallback, and min-max normalization

Status: Accepted by the project owner on 2026-11-05. Amends [ADR-0009](0009-score-normalization-and-gating.md).

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
on `t1` test, and the gain came only from users without history. B5 recent
popularity ([ADR-0016](0016-recent-popularity-baseline.md)) then reached about
four times B0 on test and beat every model in every history slice. MF, graph,
and the ADR-0009 hybrids never see recency, so they cannot show whether aspect
evidence helps.

## Decision

Add these scorer settings to `HybridBatchScorer` and
`scripts/run_hybrid_batch.py`. Each keeps its ADR-0009 value by default.

- `--cold-start fallback`: users without positive history get the baseline
  order with no aspect term and a zero gate. The baseline is B5 with
  `--cold-window-days` set to B5's tuned window, or B0 if no window is given.
  Users with history keep the hybrid formula.
- `--normalization minmax`: scale each component by the candidate minimum and
  maximum instead of percentiles. All-equal values map to 0.5.
- `--recency-window-days`: add a recency component `R`, the B5 score with
  that window, normalized like the other components.
  - H0: `w_m M + w_g G + w_r R + w_a A`, with weights that sum to 1.
  - T0: `s_base = tau R + (1 - tau)(rho M + (1 - rho) G)`; the gate is unchanged.

The ADR-0017 runs use all three settings together, with B5's tuned 60-day
window for both `R` and the cold-start fallback. They are tuned on `t0` with
`--grid adr0017`, which keeps a 9-trial budget per model:

- H0: 9 weight sets `(w_m, w_g, w_r, w_a)`, each with `w_a >= 0.1`.
- T0: `g_max` {0.1, 0.25, 0.5} x `tau` {0.4, 0.6, 0.8}, with `rho` 0.5 and
  `kappa` 5.

Report the ADR-0009 runs, the ADR-0017 runs, and B5. On `t1`, also report the
selected T0 with `g_max` 0 as an ablation: it is the same base without aspect
evidence. Do not select anything on `t1`.

## Consequences

- With the fallback, H0 and T0 equal B5 for users without history. Any
  difference from B5 then comes from users with history, where aspect evidence
  can personalize.
- ADR-0009 results stay in the record and are not overwritten.
- Aspect evidence helps only if the selected T0 beats its `g_max` 0 ablation
  and B5 on `t1`.
