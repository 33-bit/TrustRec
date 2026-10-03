# Research Claims Policy

## Allowed claims

- “The model improved NDCG@10 on this snapshot and protocol.”
- “The explanation claim matched the cited sentence in X% of audited cases.”
- “The adaptive gate helped the 1–2 interaction slice under the tested settings.”

## Disallowed without additional evidence

- Calling PageRank a reviewer honesty or fraud score.
- Calling an uncalibrated ranking score a purchase probability.
- Calling offline review-based ranking a sales, trust, or causal improvement.
- Calling AI or silver labels human gold labels.
- Treating manually labeled development/pilot data as final test data.
- Reporting a metric tuned on the frozen manual final test subset.
- Generalizing a pilot byte-window profile to the full category.

Every report paragraph that states a result must name the dataset/snapshot, cutoff, model version, seed policy, metric, and uncertainty treatment or link to the result manifest.
