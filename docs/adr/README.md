# Architecture Decision Records

ADRs record decisions. Treat accepted ADRs as immutable. If a decision changes, add a new ADR and link the old and new records. Do not change old reasoning after an experiment uses it.

ADR-0017 proposes a cold-start fallback and min-max normalization for H0 and T0. It is not accepted yet.

ADR-0016 adds B5, a recent-popularity baseline, to separate recency from aspect evidence.

ADR-0015 proposes rule-based evidence for the full corpus and leaves the LLM labeler choice open.

ADR-0014 records the sparse ranker and compact candidate design used for full-snapshot runs.

ADR-0013 changes the annotation policy. It supersedes the human-annotation parts of ADR-0007. ADR-0007 remains unchanged.
