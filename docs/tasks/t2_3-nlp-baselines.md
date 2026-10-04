# T2.3: aspect and sentiment baselines

Status: done.

This task implements a fixed dictionary baseline and a TF-IDF plus Linear SVM baseline.
The learned model uses `development_pilot` records with `llm_silver` status for fitting.
The frozen `llm_pseudo_test` records are assessment-only.
The results measure consistency with frozen LLM pseudo-labels.
They are not human agreement, ground-truth accuracy, or recommendation metrics.

The run uses snapshot `video_games-full-d6c4efeb74aa`, dataset hash `0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`, and cutoff `2019-01-13T04:24:39.377000+00:00`.
The seed is `7`.
The configuration hash is `3d02bd77ef0f48d910b15934877e4585166d00e2f3d1f52c66fe8844f2f8cee0`.
The code hash is `c63a1ccd6aa64cded4b5c118a8413d8351c19e20ed564ce37618b21ac46c9149`.
The learned model hash is `575e55f1bb121a79cfade8ee9ae0d6f8c1fed3a8eed323de07f315b651151773`.
The frozen pseudo-test labels use model `gpt-6.1-sol`, revision `unavailable`, prompt `t2.2-llm-v1`, temperature `unavailable`, and sampling seed `unavailable`.

The pilot input is `docs/tasks/t2_1_llm_review.jsonl`.
It contains 100 records.
The frozen evaluation input is `docs/tasks/t2_2_llm_pseudo_test.jsonl`.
It contains 100 records.
The learned baseline used 79 usable pilot records.
The pilot sentiment classes are negative, positive.
The learned sentiment classifier uses only these pilot classes.
Neutral pseudo-test labels remain visible in the error samples and metrics.

The dictionary baseline has aspect micro F1 0.5085 and aspect macro F1 0.4367.
Its aspect-polarity pair micro F1 is 0.4016.
Its sentiment macro F1 on detected aspects is 0.5270.
Its sentiment coverage is 0.3439.
Its evidence offset validity is 1.0000.
Its excluded-record evidence validity is 1.0000.
Its sentiment F1 on supplied reference aspect spans is 0.6392.

The TF-IDF plus Linear SVM baseline has aspect micro F1 0.0920 and aspect macro F1 0.0754.
Its aspect-polarity pair micro F1 is 0.0884.
Its sentiment macro F1 on detected aspects is 0.3333.
Its sentiment coverage is 0.0423.
Its evidence offset validity is 1.0000.
Its excluded-record evidence validity is 1.0000.
Its sentiment F1 on supplied reference aspect spans is 0.4948.

The metrics exclude 23 out-of-scope pseudo-test records and 4 records marked for review.
The run keeps refusal counts and up to 20 error samples per model.
Evidence spans remain slices of the original source text.
The learned model uses grouped cross-validation for sigmoid calibration.
It uses calibration only when each fold supports the required classes.
A calibrated probability is an estimated class probability from that step.

Run the command from the repository root:

```bash
PYTHONPATH=src python3 scripts/run_nlp_baselines.py
PYTHONPATH=src python3 scripts/validate_nlp_baselines.py
```

The completed run manifest is `docs/tasks/t2_3_nlp_baselines.manifest.json`.
It is beside the generated artifacts in `artifacts/t2_3/`.
The manifest blocks pseudo-test tuning and records artifact hashes.

Related documents are [the NLP baseline contract](../specs/nlp-baseline-contract.md), [the annotation contract](../specs/nlp-annotation-contract.md), [ADR-0013](../adr/0013-llm-only-annotation-policy.md), and [the phase 2 plan](../plans/phase-2-nlp.md).
