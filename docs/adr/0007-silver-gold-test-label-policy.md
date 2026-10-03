# ADR-0007: Silver, Gold, and Test Labels

- **Status:** Accepted
- **Date:** 2026-10-03

## Decision

All manually labeled review units are held-out evaluation data. They must not be used to train or fit the recommendation models, aspect extraction models, or sentiment models. Amazon Reviews 2023 by McAuley Lab remains the main source for training/recommendation interactions; manual labels attach evaluation truth to a sampled subset.

Manual labels have two possible roles:

1. An optional **development/pilot subset** may be used to refine the annotation guideline, prompts, thresholds, or model settings. It remains excluded from the final test set and is still excluded from model training.
2. A **final test subset** is gold-standard evaluation data for aspect extraction, sentiment classification, and explanation quality. It is frozen before tuning and cannot be used for prompt selection, model tuning, or threshold selection.

The adjudicated manual final test labels are the reference labels for those three evaluation surfaces. They do not become training labels.

Manual annotation is sampling-based; the entire Amazon Reviews 2023 corpus does not need to be labeled.

## Consequences

Every label artifact carries `label_status`, split role, and usage restrictions. Reports cannot call the current AI output human agreement or gold accuracy. T2.2 remains open until human overlap, adjudication, a frozen final test subset, and a split manifest are complete.
