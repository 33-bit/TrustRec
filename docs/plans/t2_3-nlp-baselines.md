# T2.3 implementation plan

Goal: implement dictionary and TF-IDF plus Linear SVM baselines and export frozen pseudo-label consistency results.

Architecture: preserve source offsets through clause extraction. Fit learned classifiers and calibration on development records only. Assess both fixed baselines against T2.2 after all model decisions are fixed.

Technology: Python 3.11+, scikit-learn, NumPy, PyArrow, pytest, Ruff.

Spec: `docs/specs/nlp-baseline-contract.md`, `docs/specs/nlp-annotation-contract.md`, and `docs/adr/0013-llm-only-annotation-policy.md`.

## Constraints

- Keep source text unchanged.
- Keep all fitting inputs before the exact cutoff.
- Exclude frozen pseudo-test reviews and duplicate groups from fitting.
- Keep generated model files and predictions outside Git.
- Record hashes, seed, and snapshot lineage for every result.
- Report LLM consistency with explicit assessment limits.

## Implementation steps

1. Add tests for contrast, negation, source offsets, role boundaries, and metric denominators.
2. Run `python3 -m pytest tests/unit/test_nlp_baselines.py -q` and record the missing behavior.
3. Add text extraction, dictionary rules, and input guards under `src/trustrec/nlp/`.
4. Add grouped calibration and target-conditioned sentiment with fixed parameters.
5. Add metric and error exports without reading reference labels during model fitting.
6. Add a script that validates inputs, writes immutable runs, and records hashes.
7. Run focused tests on small local fixtures without network access.
8. Run both existing label validators and the T2.3 command on the active artifacts.
9. Reproduce the run in a second output directory and compare result hashes.
10. Update the task report, backlog, module guidance, and project status.
11. Run `make check`, `make validate`, and the focused artifact validation.

## File responsibilities

`src/trustrec/nlp/text.py` owns clause boundaries and original offsets. `records.py` owns source and split guards. `dictionary.py` owns explicit rule labels. `svm.py` owns fitting and learned inference. `metrics.py` owns consistency measures. `scripts/run_nlp_baselines.py` owns run artifacts and manifests. `configs/nlp_baselines.toml` records fixed model parameters.

Tests use small records with exact source hashes and timestamps. Leakage cases include a held-out training role, a timestamp at the cutoff, overlapping reviews, and duplicate groups. Metric fixtures use hand-derived counts for missing aspects, wrong polarity, abstention, and invalid offsets.
