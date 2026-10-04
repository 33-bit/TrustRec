# T2.1: pilot status

Status: done as the LLM development pilot.

The eight-aspect ontology is ready. The repository now contains a deterministic 100-unit LLM development pilot from the full snapshot `video_games-full-d6c4efeb74aa`.

Every LLM label row has `timestamp < 2019-01-13T04:24:39.377Z`. The CSV and JSONL label records store the full source snapshot ID, dataset hash, exact cutoff, split role `development_pilot`, status `llm_silver`, source-text hash, configuration hash, model hash, unit offsets, labels, and evidence offsets in the original review text. The pilot is development data. It can refine prompts, guidelines, thresholds, and model settings. It cannot produce recommendation test metrics.

The manifest and validator link this task to [ADR-0013](../adr/0013-llm-only-annotation-policy.md), the [NLP annotation contract](../specs/nlp-annotation-contract.md), and the [phase 2 plan](../plans/phase-2-nlp.md).

Run the validator with `python3 scripts/validate_ai_pilot_labels.py`. It checks all 100 CSV rows, all 100 JSONL rows, source snapshot IDs and text, provenance parity, source-text hashes, evidence offsets, cutoff membership, and manifest hashes.

The final pseudo-test split does not exist yet. Create it from different source units after the pilot. Freeze its manifest before model or prompt selection. Use it for pseudo-label consistency and the explanation audit only.

Rebuild the worksheet with `python3 scripts/build_annotation_pilot.py --manifest data/manifests/video_games-full-d6c4efeb74aa.json --output /tmp/t2_1_annotation_pilot.csv`. To use an explicit cutoff, pass `--snapshot <snapshot-dir> --cutoff <ISO-8601-cutoff>`.

The recommendation test uses real Amazon temporal interactions and ratings. No final human gold dataset exists. Manual labeling is not required for the full source corpus.

## Files

- Guideline: `docs/tasks/t2_1-aspect-guideline.md`
- Ontology: `configs/aspects.toml`
- Pilot worksheet: `docs/tasks/t2_1_annotation_pilot.csv`
- LLM labels: `docs/tasks/t2_1_llm_review.csv`
- LLM records: `docs/tasks/t2_1_llm_review.jsonl`
- LLM manifest: `docs/tasks/t2_1_llm_review.manifest.json`
- Split template: `docs/tasks/llm-annotation-split-manifest.template.json`
- Generator: `scripts/build_annotation_pilot.py`
