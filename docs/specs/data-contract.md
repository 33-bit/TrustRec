# Data Contract

## Required tables

| Table | Required fields | Time rule |
| --- | --- | --- |
| `interactions` | `review_id`, `user_id`, `item_id`, `rating`, `timestamp` | source event time |
| `review_texts` | `review_id`, `text`, `language`, `duplicate_group` | source event time |
| `aspect_evidence` | `review_id`, `span`, `aspect`, `sentiment`, `confidence` | strictly before snapshot cutoff |
| `user_profiles` | `user_id`, `snapshot_id`, `aspect_weights`, `history_count` | built from snapshot only |
| `item_profiles` | `item_id`, `snapshot_id`, `aspect_scores`, `support` | built from snapshot only |
| `recommendations` | `user_id`, `snapshot_id`, `rank`, `item_id`, `component_scores` | serving output |
| `manual_annotation_units` | `unit_id`, `review_id`, `split_role`, `aspect`, `polarity`, `span`, `annotator_role` | held-out evaluation labels |

IDs are stable internal identifiers. Review text keeps its original content plus normalized text and offsets. Duplicate detection groups evidence but does not label fraud. Missing evidence is represented as unknown support, never as negative sentiment.

Manual annotation records must include `split_role` (`development_pilot` or `final_test`) and `label_status` (`ai_preliminary`, `human_independent`, or `adjudicated`). Both roles are held out from model training. Only the development role can influence guideline/prompt/model-setting choices; the final-test role is immutable evaluation data.
