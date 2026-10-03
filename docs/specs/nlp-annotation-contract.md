# NLP Annotation Contract

## Unit and fields

The unit is a sentence or clause with offsets into the original review text. A unit may emit multiple `(aspect, polarity, evidence span)` records. Required provenance is `review_id`, `item_id`, `timestamp`, `snapshot_id`, model/prompt version, label status, and annotator/reviewer identity type.

## Ontology

`gameplay`, `story`, `graphics`, `performance`, `controls`, `multiplayer`, `content_replay`, and `value`. Use `out_of_scope` for accessories, sellers, shipping, packaging, or unrelated hardware. Use `needs_adjudication` when entity, aspect, polarity, or span is ambiguous.

## Label policy

Use positive, negative, or neutral only for an explicit claim. Do not infer an aspect from a star rating or from generic praise unless the guideline explicitly accepts the generic gameplay interpretation. Preserve negation, contrast, and intensity. Evidence offsets must match the source string exactly.

## Data usage and split contract

The main source remains **Amazon Reviews 2023 by McAuley Lab**. Its unlabelled review/rating corpus supplies training and recommendation interactions under the temporal protocol. Manual labels are sampled held-out evaluation data and are never used to train the recommendation, aspect extraction, or sentiment models.

| Data partition | Allowed use | Forbidden use |
| --- | --- | --- |
| Amazon Reviews 2023 source corpus | recommendation interactions, snapshot construction, model fitting under cutoff | treating missing feedback as known negative |
| Manual development/pilot subset | optional guideline, prompt, threshold, and model-setting refinement; error analysis | final test scoring; model training with manual labels |
| Manual final test subset | final aspect extraction, sentiment, and explanation-quality evaluation | any tuning, prompt selection, threshold selection, or training |

AI labels are `silver`/`ai_preliminary`. The reviewed pilot is development material. The final test set is grouped by product and duplicate group, labeled by at least two annotators on the required overlap sample, frozen before tuning, and stored with an immutable split manifest. Manual labeling is not required for the entire Amazon Reviews 2023 corpus.
