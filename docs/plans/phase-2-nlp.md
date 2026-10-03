# Phase 2 Plan — Aspect NLP

**Goal:** Produce traceable aspect/sentiment evidence and a held-out manual evaluation protocol under a frozen annotation policy.

## Tasks

1. Review the 100-unit development pilot and update `configs/aspects.toml` and the guideline.
2. Keep all manually labeled units out of model training; use Amazon Reviews 2023 interactions/corpus for model fitting.
3. Lock a separate grouped manual development/final-test split after human overlap and adjudication.
4. Build an LLM/keyword silver-label pass only for development and pre-labeling.
5. Implement dictionary and TF-IDF + Linear SVM baselines with calibrated confidence without consuming manual labels for fitting.
6. Freeze the final manual test subset before prompt, threshold, or model-setting selection.
7. Export evidence offsets, duplicate groups, confidence, refusal reasons, split role, and model/prompt versions.
8. Report aspect micro/macro F1, sentiment macro F1, pair F1, span quality, explanation correctness, and coverage under abstention.

## Exit criteria

The manual final test is frozen and untouched by prompt/threshold tuning. Every prediction can be traced to a review ID and exact source span; no manual labels have entered model fitting.
