# Architecture Decision Records

ADRs record decisions. Treat accepted ADRs as immutable. If a decision changes, add a new ADR and link the old and new records. Do not change old reasoning after an experiment uses it.

ADR-0019 sets claim direction from the share of positive versus negative evidence, so popular items are not all labeled mixed.

ADR-0018 makes evidence-grounded explanation the primary contribution and treats accuracy as a guardrail against B5.

ADR-0017 amends ADR-0009: H0 and T0 get a recency term, a cold-start fallback to B5, and min-max normalization.

ADR-0016 adds B5, a recent-popularity baseline, to separate recency from aspect evidence.

ADR-0015 proposed rule-based evidence for the full corpus. ADR-0018 supersedes it: explanations use LLM evidence, and the dictionary labels remain only as a baseline.

ADR-0014 records the sparse ranker and compact candidate design used for full-snapshot runs.

ADR-0013 changes the annotation policy. It supersedes the human-annotation parts of ADR-0007. ADR-0007 remains unchanged.
