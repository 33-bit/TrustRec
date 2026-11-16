# ADR-0022: Game-only recommendation view and reselection of the explained recommender

Status: Accepted by the project owner on 2026-11-16. Amends the recommender choice in [ADR-0018](0018-explanation-first-research-focus.md).

## Context

ADR-0018 chose the ADR-0017 H0 as the recommender to explain, because it beat
B5 on all `t1` targets. The ADR-0021 game-target slice showed that this lead
comes from store credit: 59.3% of H0 top-10 items are gift cards or
memberships. On game targets, H0 scores NDCG@10 0.0043, against B3 PPR's
0.0139. TrustRec explains game recommendations, so it should explain a model
that recommends games well.

## Decision

1. **Game-only view.** Build a game-only view of the snapshot with
   `scripts/build_game_view.py`. For each cutoff, drop the ADR-0021 non-game
   items of that cutoff from interactions and targets:
   - `t0` uses the scope found with pre-`T0` labels and drops them from
     `train_interactions` and `validation_targets`;
   - `t1` uses the pre-`T1` scope and drops them from `fit_interactions` and
     `test_targets`.

   Non-game items leave histories, the candidate catalog, and targets. The
   view has its own snapshot ID (`<id>-games`), dataset hash, and artifact
   hashes. The full-catalog results stay in the record.
2. **Tuning on `t0`.** Every tuned component is tuned again on `t0` of the
   view, with the same budgets as before:
   - B5 window: 9 trials.
   - BPR MF: refit with the earlier selected settings (learning rate 0.01,
     regularization 1e-4, 30 epochs, seed 7). It gets no new grid, because
     H0 and T0 weight it.
   - B3 PPR: damping 0.85 and 100 iterations, the earlier winner.
   - H0 and T0 (`--grid adr0022`, 9 trials each): normalization, recency, and
     the cold-start fallback follow ADR-0017. The grid covers low and zero
     recency weights, because recency helped mainly through store credit.
3. **Cold-start fallback.** Users without positive history get B5 if B5
   beats B0 on `t0` of the view; otherwise they get B0.
4. **Selection.** The explained recommender is whichever of H0 and T0 scores
   better on `t0` validation NDCG@10. Test on `t1` once. The ADR-0018 accuracy
   guardrail now compares against the strongest baseline on `t0` of the view,
   not against B5.
5. **Explanations.** Regenerate them for the selected recommender on `t1` of
   the view. Then verify the display passages (ADR-0020), with non-game items
   already removed (ADR-0021).

## Consequences

- Game-only and full-catalog results are separate tables with separate
  snapshot IDs. They must not be mixed in one comparison.
- The ADR-0018 claim that H0 is "better than B5" refers to the full catalog
  only.
