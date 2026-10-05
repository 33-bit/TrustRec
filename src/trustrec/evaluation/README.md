# Evaluation Module

This module builds temporal evaluation cases and validates shared candidate
and target sets. It owns leakage checks, ranking metrics, data slices, paired
user bootstrap intervals, explanation coverage, and run-manifest fields.

The runner reads prepared per-user ranking JSON Lines files. It does not fit a
model or download source data. Keep test data read-only after protocol lock.
