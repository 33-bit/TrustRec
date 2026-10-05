# Explanations Module

This module selects evidence and writes claims. It also aggregates one
snapshot of aspect evidence. The aggregator applies confidence, duplicate,
and optional recency weights, then stores score, support, sentiment mass,
author count, and effective sample size. It rejects evidence at or after the
requested cutoff and keeps conflicting sentiment visible.

T4.2 selects claims by user aspect weight and item deviation from the prior.
It stores exact review offsets, support limits, score parts, and refusal
reasons. The removal audit keeps candidate normalization fixed and records
original, cited-removal, and random-removal contributions.
