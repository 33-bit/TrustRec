# Annotation Review Protocol

1. Keep Amazon Reviews 2023 by McAuley Lab as the main source; manually label a sample, not the whole corpus.
2. Freeze the unit list and source offsets before label review.
3. Assign every manual unit a split role: `development_pilot` or `final_test`.
4. Keep the AI label visible only as a proposal; annotators record their own decision.
5. Fill aspect/polarity/evidence span, out-of-scope, adjudication flag, and rationale.
6. At least 20% of units receive two independent labels.
7. Adjudicate disagreements against the guideline and record the decision reason.
8. Keep all manual labels out of recommendation, aspect-extraction, and sentiment-model training.
9. Use development labels only for guideline/prompt/model-setting refinement; never use them as final test.
10. Freeze the final test subset before any tuning or prompt selection; use it only for final aspect, sentiment, and explanation evaluation.
11. Export a manifest with annotator roles, date, guideline version, source snapshot, split role, and usage restrictions.
