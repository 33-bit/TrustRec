# NLP annotation contract

An annotation record stores an LLM label for one sentence or clause. The record keeps the original review text, the unit text, and the unit offsets. Evidence offsets refer to the original review text.

## Source and split roles

Amazon Reviews 2023 by McAuley Lab remains the source. The source reviews and ratings provide recommendation interactions and data for model fitting under the temporal protocol. The project does not require manual labels for the full corpus.

| Role | Allowed use | Forbidden use |
| --- | --- | --- |
| `development_pilot` with `llm_silver` | Refine guidelines, prompts, thresholds, model settings, optional NLP training, NLP features, and evidence. | Recommendation test scoring. |
| `llm_pseudo_test` with `llm_pseudo_test` | Frozen aspect, sentiment, evidence, and explanation consistency checks. | Prompt, threshold, model, or hyperparameter tuning after freeze. |
| `recommendation_test` | Recommendation metrics on real Amazon user-item interactions and ratings. | Use as a source of NLP labels. |

LLM labels are not human gold labels. Do not report human agreement, Cohen kappa, or ground-truth accuracy. Report pseudo-label agreement or consistency. Keep recommendation metrics separate from NLP checks.

## Label rules

Use these aspects: `gameplay`, `story`, `graphics`, `performance`, `controls`, `multiplayer`, `content_replay`, and `value`. Use `out_of_scope` for accessories, consoles, controllers, cables, headsets, sellers, shipping, packaging, and unrelated hardware.

Use `positive`, `negative`, or `neutral` only for an explicit claim. Keep negation, contrast, and intensity. Evidence text must equal the source text at the recorded offsets. Mark an unclear unit for model abstention or error review. Do not invent a label from a star rating.

## Required provenance

Each record stores schema version 2, `unit_id`, `review_id`, `item_id`, `source_snapshot_id`, `timestamp`, the snapshot dataset hash, the exact ISO-8601 `source_cutoff_timestamp`, `split_role`, `label_status`, `model_id`, `model_revision`, `prompt_version`, `temperature`, `seed`, `generated_at`, `source_text_sha256`, `unit_start`, `unit_end`, `unit_text`, the configuration hash, the model hash, labels, evidence offsets, scope flags, and usage restrictions. The schema enforces these fields.

When a model session does not expose its revision, temperature, or sampling seed, record `unavailable`. Keep the split selection seed separate. Do not invent a provider revision or sampling value.

Freeze the pseudo-test manifest before model or prompt selection. Keep the frozen test data unchanged. Archive old pilot outputs when their source snapshot or provenance does not match the active manifest.
