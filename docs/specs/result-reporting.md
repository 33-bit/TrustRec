# Result reporting contract

Each result table names the model ID, snapshot ID, eligible-user count, metric, seed summary, and uncertainty method. Each figure names its units and candidate and target rules.

Separate measured results, design claims, and limits. Do not choose a model after reading test metrics.

## Primary results: explanations

Under [ADR-0018](../adr/0018-explanation-first-research-focus.md), explanation results come first. Report them for the top-K of the explained recommender (the ADR-0017 H0) on the real snapshot:

- Coverage: the share of recommended items with at least one `supported` claim, and the share of users with at least one explained item.
- Abstention: the shares of `insufficient_support` and `unavailable`.
- Faithfulness: the score change when cited evidence is removed, compared with a random removal of the same size.
- Audit: item correctness, aspect correctness, sentiment direction, evidence traceability, and abstention. When an LLM judge is used, record its model and prompt, and report its results as LLM agreement with known limits.

Group coverage and abstention by history slice, item popularity, and item age. Name the evidence source, which is the curated LLM labels under ADR-0018.

## Accuracy guardrail

Report the explained recommender against B5 on `recommendation_test` interactions, with a paired bootstrap interval of the NDCG@10 difference. Use "on par with B5" only when the whole 95% interval lies within ±0.005, and "better than B5" only when it lies above 0. Also report B0 to B3, the ADR-0009 hybrids, and the ablations as context, with NDCG@K, Recall@K, Precision@K, coverage, diversity, popularity slices, and sparse-history slices. Do not claim that aspect evidence improves accuracy.

## NLP consistency

Report NLP results from `llm_pseudo_test` as aspect, sentiment, and evidence consistency. State the LLM model, revision, prompt version, seed, and source snapshot. Do not call these results human agreement, gold accuracy, or ground-truth accuracy. Keep them separate from recommendation and explanation results.
