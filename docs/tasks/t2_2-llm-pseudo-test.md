# T2.2: frozen LLM pseudo-test

Status: done.

The project now has a frozen `llm_pseudo_test` split with 100 source reviews. The split uses the full snapshot `video_games-full-d6c4efeb74aa`.

The source dataset hash is `0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`. The exact cutoff is `2019-01-13T04:24:39.377000+00:00`. Every selected review has a timestamp before this cutoff.

The selection excludes all 100 `development_pilot` review IDs and their `item_id` plus `duplicate_group` pairs. It keeps 20 reviews from each rating bucket. The selection seed is `2202`. The minimum source text length is 100 characters.

The current Codex model generated 189 aspect labels. The records include 23 out-of-scope units and 4 units that need review. Evidence strings match exact source offsets. The pass uses the prompt version `t2.2-llm-v1`.

The manifest records model ID `gpt-6.1-sol`. The Codex desktop session did not expose the provider revision, temperature, or sampling seed. The manifest records each unavailable value as `unavailable`. The model hash covers all recorded model fields and the prompt version.

The model hash is `db9fe930b0a839d201704293d88c400a99c6c1bb3dcef3dfda0df38dd4a445fb`. The configuration hash is `1569c6f65cca126b6ec41ce195b4a3ea5842a7d1e23c5129eb4c3a9241298238`. The development pilot artifact hash is `d8c336c8d321bc98b19d1906cbcaf5d44f65d5689ba396f5908e3553cda6ce5d`.

The frozen manifest blocks prompt, threshold, model, and hyperparameter tuning. The validator also checks the source artifact hashes, source cutoff, 100-unit count, rating quotas, source text hashes, evidence offsets, snapshot IDs, duplicate-group overlap, and pilot overlap.

Run the validator from the repository root:

```bash
python3 scripts/validate_llm_pseudo_test.py \
  --labels docs/tasks/t2_2_llm_pseudo_test.csv \
  --jsonl docs/tasks/t2_2_llm_pseudo_test.jsonl \
  --manifest docs/tasks/t2_2_llm_pseudo_test.manifest.json
```

The validator reported 100 validated units. It reported split role `llm_pseudo_test`, model ID `gpt-6.1-sol`, and prompt version `t2.2-llm-v1`.

## Files

- Manifest: `docs/tasks/t2_2_llm_pseudo_test.manifest.json`
- CSV records: `docs/tasks/t2_2_llm_pseudo_test.csv`
- JSONL records: `docs/tasks/t2_2_llm_pseudo_test.jsonl`
- Source selection: `docs/tasks/t2_2_llm_pseudo_test_selection.jsonl`
- Locked prompt: `docs/tasks/t2_2-llm-pseudo-test-prompt.md`
- Selection script: `scripts/build_llm_pseudo_test.py`
- Validator: `scripts/validate_llm_pseudo_test.py`

The records are LLM pseudo-labels. They are not human gold labels. Use them for aspect, sentiment, evidence, and explanation consistency checks. Keep recommendation metrics on the separate `recommendation_test` interactions.
