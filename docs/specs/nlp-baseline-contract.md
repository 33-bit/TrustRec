# NLP baseline contract

T2.3 produces aspect and sentiment predictions, consistency metrics, and error samples. A baseline is a model used for comparison.

## Inputs and isolation

Use the T2.1 `development_pilot` records with `llm_silver` status for model fitting. Use T2.2 frozen `llm_pseudo_test` records only for assessment.

Each input must contain its source snapshot, exact cutoff, text hash, and evidence offsets. All source timestamps must precede the cutoff. Reject overlap in review IDs and item-scoped duplicate groups between training and assessment.

Do not fit a vocabulary, model, calibration function, or threshold on the pseudo-test. Calibration maps model scores to estimated class probabilities. Fit calibration through grouped cross-validation on the development data. Cross-validation fits separate models on partitions of the training data.

Keep clauses from the same review and duplicate group together during calibration. Each fold must contain every fitted class in both partitions. Refuse unsupported classifiers when the training data cannot supply these folds.

## Models

The dictionary baseline uses the eight existing aspects, sentiment terms, local negation, and clause boundaries. It refuses claims without explicit sentiment cues. Its confidence is a rule score, not a probability.

The learned baseline uses TF-IDF and Linear SVM. TF-IDF weights words by their frequency and rarity. Linear SVM separates labels through a fitted linear decision boundary.

Fit one binary classifier per aspect. Fit sentiment on clause text with target aspect features. Use calibrated probabilities for learned confidence. Record absent sentiment classes and unsupported aspect classifiers.

Fix seed 7, word n-grams 1 and 2, `C = 1.0`, three calibration folds, aspect threshold 0.5, and sentiment threshold 0.6. Use these values before pseudo-test assessment. No parameter search occurs in T2.3.

## Predictions

Keep the source text unchanged. Export each label with aspect, polarity, confidence, sentiment score, and exact character offsets. An offset identifies a position in the original text.

Store refusal reasons for unsupported scope, missing aspects, uncertain sentiment, and unsupported training classes. Keep predictions independent of reference labels. Produce separate target-conditioned sentiment predictions for assessment on supplied reference aspects.

The main prediction files use schema version 2 provenance fields. The target-conditioned files use schema version 1 for one supplied reference aspect span per row. They store the target aspect, target polarity, target offsets, target evidence, predicted polarity, confidence, refusal reason, snapshot fields, model hash, configuration hash, and seed.

## Assessment

Report aspect micro and macro F1, aspect-polarity pair F1, and sentiment macro F1 on supplied reference aspects. F1 combines precision and recall. Also report sentiment consistency on detected aspects, prediction coverage, and abstention counts.

Quality metrics exclude reference units that need review or are out of scope. Report their counts and false evidence separately. Measure exact offset validity and reference evidence containment with explicit numerators and denominators.

Export bounded error samples after model parameters are fixed. Do not use these samples to tune the model in this task.

Export separate error samples for eligible records and excluded records. Excluded records remain outside quality scores, but their evidence checks remain visible.

## Artifacts

Keep generated predictions and model files in ignored `artifacts/`. Store the task report and small metrics manifest in `docs/tasks/`. Each run records dataset hash, snapshot ID, cutoff, seed, input hashes, configuration hash, code hash, model hash, and artifact hashes.

Store the learned model as a stable JSON artifact that includes the fitted vocabulary, IDF values, classifier parameters, calibration state, and model hash. Do not overwrite a completed run. Use a new run ID for another result. Repeated runs with the same inputs must produce matching prediction, model, and metric hashes.

This assessment measures agreement with LLM pseudo-labels. It does not measure human agreement or recommendation quality.
