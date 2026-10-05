# Explanations Module

This module selects evidence and writes claims. It also aggregates one
snapshot of aspect evidence. The aggregator applies confidence, duplicate,
and optional recency weights, then stores score, support, sentiment mass,
author count, and effective sample size. It rejects evidence at or after the
requested cutoff and keeps conflicting sentiment visible.

Claim selection and faithfulness checks belong to T4.2. T4.1 does not invent
evidence or hide a dominant non-aspect score.
