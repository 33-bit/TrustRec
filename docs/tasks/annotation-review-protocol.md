# LLM annotation protocol

This protocol uses Amazon Reviews 2023 by McAuley Lab. It does not require human labeling.

1. Build the unit list from a snapshot manifest or an explicit cutoff.
2. Keep only reviews with `timestamp < T0`.
3. Store the exact ISO-8601 cutoff, snapshot ID, snapshot dataset hash, item ID, review ID, and source-text hash.
4. Run the LLM with a recorded model ID, model revision, prompt version, temperature, seed, configuration hash, and model hash.
5. Store aspect labels, sentiment, evidence text, unit offsets, and evidence offsets in the original review.
6. Mark development output as `split_role = development_pilot` and `label_status = llm_silver`.
7. Use the development pilot for prompt, guideline, threshold, and model-setting work.
8. Create an independent `llm_pseudo_test` pass after development work ends.
9. Freeze the pseudo-test manifest before model or prompt selection.
10. Use the frozen pseudo-test only for aspect, sentiment, evidence, and explanation consistency.
11. Use `recommendation_test` interactions and ratings for recommendation metrics.
12. Store allowed and prohibited uses in every record and manifest.

Do not call LLM labels human gold labels. Do not report human agreement or Cohen kappa. Do not use the frozen pseudo-test for tuning. Do not label the full Amazon Reviews 2023 corpus.
