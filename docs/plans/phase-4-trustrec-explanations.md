# Phase 4 Plan: TrustRec and Explanations

Combine ranking signals and return explanations with evidence. Evidence is source text that supports a claim. Each explanation links its claim to a score contribution.

## Tasks

1. Aggregate evidence with confidence, duplicate penalties, optional recency, shrinkage, support, author counts, and effective sample size.
2. Read `docs/specs/evidence-aggregation-contract.md` for the meaning of each weight and support measure.
3. Build user aspect weights from pre-cutoff history with smoothing.
4. Add editable aspect priorities to the UI.
5. Compare H0 with T0 under equal validation budgets.
6. Create claims from score contributions and cited evidence.
7. Add wording for weak support, conflicting evidence, and dominant score parts.
8. Run evidence-removal tests for faithfulness.
9. Store original and recomputed contributions.

## Exit rules

Each recommendation has score parts and an explanation status. Strong claims meet support thresholds selected on validation. Measure refused explanations too.
