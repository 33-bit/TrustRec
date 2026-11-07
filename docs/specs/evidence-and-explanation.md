# Evidence and Explanation Contract

An explanation claim names the item and aspect. Sentiment states whether the opinion is positive, negative, or neutral. Link each claim to one or more `EvidenceRef` records.

Distinguish positive support, negative trade-offs, weak support, and conflicting support. [ADR-0019](../adr/0019-claim-direction-by-evidence-share.md) sets the direction from the positive share of polar evidence mass: at least 2/3 is positive, at most 1/3 is negative, and between is mixed. Positive and negative claims state the share and cite one passage from the other side when one exists. Templates can say “review evidence before the cutoff”. Do not say “most users agree” without an explicit numerator, denominator, review set, and time window.

The service returns `supported`, `insufficient_support`, or `unavailable`. Select thresholds on validation data. Strong claims require enough distinct authors, effective support, and NLP confidence.

Faithfulness measures whether cited evidence affects a score. Remove that evidence and recompute the local score. Compare the change with a random removal of the same size.
