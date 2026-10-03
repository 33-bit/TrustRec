# Evidence and Explanation Contract

An explanation claim names the item and aspect, states the sentiment direction, and links to one or more `EvidenceRef` records. It must distinguish positive support, negative trade-offs, and insufficient or conflicting support. Templates may say “review evidence before the cutoff” but must not say “most users agree” unless numerator, denominator, review set, and time window are explicit.

The explanation service returns `supported`, `insufficient_support`, or `unavailable`. A strong explanation requires validation-selected thresholds for distinct authors, effective support, and NLP confidence. Faithfulness checks recompute the local score after removing cited evidence and compare it with a same-size random removal.

