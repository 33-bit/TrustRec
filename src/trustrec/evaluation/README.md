# Evaluation Module

This module builds temporal evaluation cases and validates shared candidate
and target sets. It owns leakage checks, ranking metrics, data slices, paired
user bootstrap intervals, explanation coverage, and run-manifest fields.

The runner reads prepared per-user ranking JSON Lines files. It does not fit a
model or download source data. Keep test data read-only after protocol lock.

Protocol `evaluation-v2` keeps one shared catalog and each user's history
instead of a full candidate list per user. A ranking row declares its pool
with `candidate_item_ids`, or with `candidate_rule` set to
`catalog_minus_history`, `candidate_count`, and `candidate_signature`. The
signature comes from `trustrec.recommenders.matrix.candidate_pool_signature`.
Every ranked item must be in the catalog and outside the user's history.
