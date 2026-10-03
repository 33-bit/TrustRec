# T2.1 — Pilot Status

The eight-aspect ontology and annotation guideline are ready. A deterministic worksheet of 100 unique review units was generated from snapshot `video_games-pilot-9e665a862c1a` with rating quotas 20/15/15/25/25 for ratings 1–5.

The worksheet is explicitly a `development_pilot` subset of manually labeled held-out evaluation data. It contains weak keyword suggestions only. Both annotator columns and the adjudicated column are empty by design. Its labels may refine the guideline, prompt, or model settings, but they must not enter recommendation, aspect-extraction, or sentiment-model training and cannot become the final test set. T2.1 cannot be marked complete until two independent annotations cover at least 20% of the pilot and disagreements are resolved against the guideline. T2.2 must wait for those labels before computing agreement or locking the frozen final test split.

One subagent also produced an AI-only preliminary artifact from 79 pre-`T0` source reviews: 100 sentence/clause units with exact evidence offsets. It contains 52 out-of-scope units, 38 needs-adjudication flags, and 67 units without an ontology label. It is support for human review, not gold data or an independent annotator.

Laya local labeling is also available in `docs/tasks/t2_1_laya_pilot_labels.jsonl` with a reviewer-friendly CSV at `docs/tasks/t2_1_laya_review.csv`. The run is conservative at presence threshold 0.85 and remains AI preliminary; its runtime warning makes confidence uncalibrated for this pilot.

Files:

- Guideline: `docs/tasks/t2_1-aspect-guideline.md`
- Ontology: `configs/aspects.toml`
- Worksheet: `docs/tasks/t2_1_annotation_pilot.csv`
- LLM review CSV: `docs/tasks/t2_1_llm_review.csv`
- LLM review manifest: `docs/tasks/t2_1_llm_review.manifest.json`
- Generator: `scripts/build_annotation_pilot.py`
- AI preliminary labels: `docs/tasks/t2_1_ai_pilot_labels.jsonl`
- AI label report: `docs/tasks/t2_1-ai-labeling-report.md`
- Laya labels: `docs/tasks/t2_1_laya_pilot_labels.jsonl`
- Laya review CSV: `docs/tasks/t2_1_laya_review.csv`
