# T2.1: LLM development pilot report

Status: done as the development pilot. Related records are [ADR-0013](../adr/0013-llm-only-annotation-policy.md), the [NLP annotation contract](../specs/nlp-annotation-contract.md), and the [phase 2 plan](../plans/phase-2-nlp.md).

Data role: `development_pilot` with status `llm_silver`.

The 100 units come from reviews before `T0 = 2019-01-13T04:24:39.377Z` in full snapshot `video_games-full-d6c4efeb74aa`. The snapshot dataset hash is `0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`. The pilot is separate from the recommendation test and is not a human gold set.

Each row stores the original review text, the labeled unit text and offsets, the source-text SHA-256 hash, review ID, item ID, timestamp, labels, evidence offsets, and LLM provenance. Evidence offsets use the original review text. The records also store the configuration hash `f03fb8709368efbb1f2c7cf6f3302450d111ba5b3099eaaf2a9485e64f6a641e` and model hash `e7410b06c2bfd4fd91bd46e8cec94ef97b1ac6e72807389d43df11211e6ae63c`. The manifest defines the hash inputs. The model hash identifies the recorded model and run settings. It does not identify model weights.

Use this pilot to find prompt and ontology errors. You can change the prompt, guideline, threshold, or model settings after reviewing this data. You can use its labels for optional NLP training. Do not use the pilot for final pseudo-test scoring or recommendation metrics.

Rebuild the source worksheet with `python3 scripts/build_annotation_pilot.py --manifest data/manifests/video_games-full-d6c4efeb74aa.json --output /tmp/t2_1_annotation_pilot.csv`. Run `python3 scripts/validate_ai_pilot_labels.py` to validate both label artifacts. The validator checks 100 CSV units, 100 JSONL units, exact provenance parity, source snapshot review IDs and text, source-text hashes, evidence offsets, cutoff membership, and manifest hashes.

Create the independent `llm_pseudo_test` split after development work ends. Freeze it before tuning. Report its results as aspect, sentiment, evidence-span, or explanation consistency. Do not call the results human agreement or ground-truth accuracy.
