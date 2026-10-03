# T2.1 AI Preliminary Labeling Report

**Data role:** development/pilot only. These AI labels are held out from all model training and may support guideline/prompt/model-setting refinement. They are not final-test labels.

## Scope and selection

This artifact contains 100 sentence/clause units selected deterministically from the existing pilot worksheet. The manifest cutoff **T0 is 2021-04-09T19:03:26.064Z**; only worksheet reviews with `timestamp < T0` were eligible, preventing post-T0 target text from entering corpus-development labels. There were **79 eligible source reviews**. Units were extracted from the raw worksheet text with Python character offsets and selected in round-robin passes over source review order (one unit per review per pass), yielding 100 unique `(review_id, start, end)` units across all 79 eligible reviews.

Selection and offset validation are reproducible with `python3 scripts/validate_ai_pilot_labels.py`.

## Artifact and provenance

- `docs/tasks/t2_1_ai_pilot_labels.jsonl` — exactly 100 records, version `t2.1-ai-v1`.
- Every record is marked `ai_preliminary` with provenance `subagent-label-pilot`.
- Labels were manually reasoned from each selected text unit; weak keyword suggestions and a classifier were not used. Original review text, stable IDs, timestamps, SHA-256 source text hashes, and evidence offsets are retained.
- This is not a gold set, independent annotator output, or adjudicated data. Human annotator columns in the worksheet remain untouched.

## Validation summary

| Check | Result |
| --- | ---: |
| Unique units | 100 |
| Pre-T0 source reviews | 79 |
| Out-of-scope hardware/merchant units | 52 |
| Needs adjudication | 38 |
| Units without an ontology label | 67 |

Evidence spans were checked to match the original source string exactly at their recorded offsets. Label counts are: gameplay positive 13, gameplay negative 2, story negative 1, graphics positive 2, graphics negative 1, performance negative 8, controls negative 1, multiplayer positive 1, content/replay positive 1, content/replay negative 2, and value positive 4.

Many worksheet entries describe accessories, packaging, shipping, or very short/vague opinions. They are retained with `out_of_scope` or `needs_adjudication` flags rather than forcing a game-aspect interpretation. Human review is still required before using any labels as gold data.
