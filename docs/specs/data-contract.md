# Data contract

A contract defines the fields that modules exchange. A cutoff is the time boundary for data use. Feature and evidence rows must contain data before that cutoff.

| Table | Required fields | Time rule |
| --- | --- | --- |
| `interactions` | `review_id`, `user_id`, `item_id`, `rating`, `timestamp` | Source event time. |
| `review_texts` | `review_id`, `item_id`, `text`, `timestamp`, `source_text_sha256`, `duplicate_group` | Source event time. |
| `aspect_evidence` | `review_id`, `item_id`, `span`, `aspect`, `sentiment`, `confidence`, `source_timestamp` | Before the snapshot cutoff. |
| `user_profiles` | `user_id`, `snapshot_id`, `aspect_weights`, `history_count` | Snapshot data only. |
| `item_profiles` | `item_id`, `snapshot_id`, `aspect_scores`, `support` | Snapshot data only. |
| `recommendations` | `user_id`, `snapshot_id`, `rank`, `item_id`, `component_scores` | Serving output. |
| `llm_annotation_units` | `unit_id`, `review_id`, `item_id`, `source_snapshot_id`, `source_cutoff_timestamp`, `split_role`, `label_status`, model and prompt provenance, labels, evidence offsets, usage restrictions | Source review time and exact cutoff. |

IDs are stable internal identifiers. Keep original text, normalized text, and offsets. A duplicate group hashes normalized text within one item. It does not label fraud. Missing evidence means unknown support.

## Annotation roles

Use `development_pilot` with `llm_silver` for prompt work, optional NLP training, features, and evidence. Use frozen `llm_pseudo_test` with `llm_pseudo_test` for pseudo-label consistency checks. Use `recommendation_test` for real Amazon temporal interactions and ratings. Do not call LLM labels human gold labels.

Do not use the frozen pseudo-test for tuning. Do not use annotation records as recommendation test interactions. Manual labeling is not required for the Amazon Reviews 2023 corpus.
