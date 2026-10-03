# Phase 4 Plan — TrustRec and Explanations

**Goal:** Combine ranking signals and return evidence-aware explanations.

## Tasks

1. Aggregate review evidence with confidence, duplicate penalty, optional recency, shrinkage, support, authors, and effective sample size.
2. Build user aspect weights from pre-cutoff history with smoothing and editable UI priorities.
3. Compare fixed hybrid H0 with adaptive gate T0 under equal validation budgets.
4. Generate explanation claims from score contributions and cited evidence.
5. Add insufficient-support, conflict, and component-dominance wording.
6. Run evidence-removal faithfulness tests and persist original/recomputed contributions.

## Exit criteria

Every recommendation has score components and an explanation status. Strong claims require validation-selected support thresholds; refused explanations remain measurable.

