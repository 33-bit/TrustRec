# T2.1 — Laya Local Run Report

**Status:** AI preliminary labels generated; human review pending.  
**Data role:** development/pilot held-out evaluation support; these labels are not used to train recommendation, aspect extraction, or sentiment models.  
**Model:** `laya` English checkpoint, package `laya==0.3.23`  
**Input:** 100 pre-T0 sentence/clause units from 79 source reviews  
**Presence threshold:** `0.85`  
**Execution:** local Python process on CPU; first run downloaded the checkpoint from Hugging Face, then reused the local cache.

## Output

- JSONL: `docs/tasks/t2_1_laya_pilot_labels.jsonl`
- Reviewer CSV: `docs/tasks/t2_1_laya_review.csv`
- Runner: `scripts/run_laya_labeling.py`
- CSV exporter: `scripts/export_laya_review_csv.py`

Summary: 100 units, 12 emitted aspect/polarity labels, 7 out-of-scope units, 43 units flagged for human review, and 95 units with no emitted label at the 0.85 presence threshold. The high abstention rate is acceptable for a conservative pilot and should be reviewed before changing the threshold.

## Important limitation

The Laya checkpoint emitted a runtime warning that one shipped temperature value was outside the supported range and was clamped. Treat the stored confidence values as **uncalibrated for this run**. They are useful for triage, not as validated probabilities. The model also returns decisions rather than evidence spans, so each output uses the selected sentence/clause as its evidence span.

These outputs are `ai_preliminary` silver labels. They must not be reported as human agreement, gold labels, or final NLP accuracy. Review the CSV columns `labels_json`, `out_of_scope`, `needs_adjudication`, and `human_labels_json`; add human decisions only in the blank human columns.
