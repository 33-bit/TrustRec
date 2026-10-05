# Explanation audit contract

The explanation audit checks claims attached to recommendations. It does not replace recommendation metrics.

## Audit fields

Each audit row stores the recommendation ID, user ID, item ID, snapshot ID, claim, aspect, sentiment direction, evidence review ID, evidence offsets, support status, and audit result.

Check these properties:

- Item correctness. The explanation names the recommended item.
- Aspect correctness. The claim uses an aspect supported by the evidence.
- Sentiment direction. The claim matches the evidence polarity.
- Evidence traceability. The evidence points to a source review and exact text offsets.
- Abstention. The system refuses a claim when support is insufficient.

An LLM judge is optional. If the project uses one, record its model ID, model revision, prompt version, temperature, seed, and generation time. Report the result as LLM agreement or consistency. Do not report it as human judgment or ground-truth accuracy.

The NLP pseudo-test checks aspect, sentiment, and evidence-span consistency. The recommendation test uses real Amazon temporal interactions and ratings. Keep both result types in separate manifests.

The implementation stores the cited review IDs and character offsets in each
audit row. It stores the original, cited-removal, and random-removal score
parts. The random removal uses a fixed seed and removes the same number of
review IDs as the cited removal. The candidate set hash is recorded when the
caller supplies candidate IDs. The normalization path remains fixed for every
recomputed score.

The audit artifact follows [`explanation-audit.schema.json`](../../schemas/explanation-audit.schema.json).
