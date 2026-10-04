# NLP Module

This module splits reviews into clauses and preserves offsets in the original text. It provides a fixed dictionary baseline and a TF-IDF plus Linear SVM baseline for aspect and sentiment prediction.

The learned baseline fits only `development_pilot` records with status `llm_silver`. It keeps the frozen `llm_pseudo_test` split for consistency checks. It uses grouped cross-validation for sigmoid calibration when the development data supports the required classes.

Every prediction keeps the source snapshot, cutoff, text hash, model hash, configuration hash, label offsets, confidence kind, and refusal reason. A calibrated probability is an estimated class probability. A rule score is not a probability.

Run the baselines from the repository root:

```bash
PYTHONPATH=src python3 scripts/run_nlp_baselines.py
```

Read the [T2.3 report](../../docs/tasks/t2_3-nlp-baselines.md) and [baseline contract](../../docs/specs/nlp-baseline-contract.md) for the fixed settings, split limits, metrics, and artifact paths.
