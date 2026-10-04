# Research claims policy

State only claims supported by a recorded manifest and experiment.

## Do not claim

- Call LLM labels human gold labels.
- Call pseudo-label consistency human agreement.
- Report Cohen kappa for LLM outputs.
- Treat an LLM label as ground truth.
- Use the frozen pseudo-test for tuning.
- Treat the development pilot as the final pseudo-test.
- Treat a filtered pilot as a full-category benchmark.
- Treat an Amazon rating or interaction as an aspect label.

Each result names its source dataset, snapshot, cutoff, model, prompt version, seed, metric, and uncertainty method. Recommendation results use the real temporal test interactions. NLP and explanation results use LLM consistency language.
