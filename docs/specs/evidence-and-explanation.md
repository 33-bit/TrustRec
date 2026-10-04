# Evidence and Explanation Contract

An explanation claim names the item and aspect. Sentiment states whether the opinion is positive, negative, or neutral. Link each claim to one or more `EvidenceRef` records.

Distinguish positive support, negative trade-offs, weak support, and conflicting support. Templates can say “review evidence before the cutoff”. Do not say “most users agree” without an explicit numerator, denominator, review set, and time window.

The service returns `supported`, `insufficient_support`, or `unavailable`. Select thresholds on validation data. Strong claims require enough distinct authors, effective support, and NLP confidence.

Faithfulness measures whether cited evidence affects a score. Remove that evidence and recompute the local score. Compare the change with a random removal of the same size.
