# TrustRec terms

| Term | Meaning |
| --- | --- |
| Aspect | A product property such as gameplay or performance. |
| Evidence | Source text that supports a recommendation claim. |
| Cutoff | The time boundary for data use. |
| Snapshot | A versioned view of source data and its cutoff. |
| `llm_silver` | LLM labels for development, features, or evidence. |
| `llm_pseudo_test` | Frozen LLM labels for pseudo-label consistency checks. |
| `recommendation_test` | Real Amazon temporal interactions and ratings for recommendation metrics. |
| Development pilot | A split that can refine prompts, guidelines, thresholds, or model settings. |
| Pseudo-label consistency | Agreement between a model output and a frozen LLM label pass. |
| Abstention | A refusal to emit a claim when support is insufficient. |
| Duplicate group | An item-scoped hash for repeated normalized review text. |
| Leakage | Future or held-out information that affects a prediction. |
| Coverage | The fraction of eligible cases that receive a result. |

LLM labels are not human gold labels. The project does not use human annotator agreement or Cohen kappa. Manual labeling is not required for the full Amazon Reviews 2023 corpus.
