# Result reporting contract

Each result table names the model ID, snapshot ID, eligible-user count, metric, seed summary, and uncertainty method. Each figure names its units and candidate and target rules.

Separate measured results, design claims, and limits. Do not choose a model after reading test metrics.

Report NLP results from `llm_pseudo_test` as aspect, sentiment, and evidence consistency. State the LLM model, revision, prompt version, seed, and source snapshot. Do not call these results human agreement, gold accuracy, or ground-truth accuracy.

Use the explanation audit for item correctness, aspect correctness, sentiment direction, evidence traceability, and abstention. Record an LLM judge model and prompt when a judge is used. Report judge results as LLM agreement with known limits.

Report recommendation results from `recommendation_test` interactions. Include NDCG@K, Recall@K, Precision@K, coverage, diversity, popularity slices, and sparse-history slices. Keep this report independent from LLM label consistency.
