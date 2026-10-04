# Phase 2 plan: aspect NLP

Create aspect and sentiment evidence from Amazon Reviews 2023. Use LLM labels as development data or frozen pseudo-test data. Do not describe them as human gold labels.

## Tasks

1. Run the 100-unit `development_pilot` before the exact `T0` cutoff.
2. Record source snapshot, cutoff, model revision, prompt version, seed, and text hashes.
3. Refine the ontology and prompt with the development pilot only.
4. Freeze an independent `llm_pseudo_test` split and lock its manifest.
5. Keep the frozen pseudo-test out of prompt, threshold, model, and hyperparameter tuning.
6. Build dictionary and TF-IDF plus Linear SVM baselines from Amazon source data.
7. Export labels, evidence offsets, refusal reasons, split roles, and provenance.
8. Report aspect, sentiment, and evidence-span consistency.
9. Run the explanation audit with separate item, aspect, sentiment, traceability, and abstention checks.

## Exit rules

The pilot is development-only. The pseudo-test manifest is frozen. The recommendation test remains a temporal Amazon interaction split. Every result names its data role and does not claim human accuracy.
