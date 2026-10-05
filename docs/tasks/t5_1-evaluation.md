# T5.1: Baseline, ablation, and sparse-history evaluation

Status: Done for the evaluation runner and deterministic fixture. A full
Video Games result requires the external Parquet tables and ranking artifacts.

## Related records

- Backlog: [`backlog.md`](backlog.md)
- Evaluation contract: [`../specs/evaluation-contract.md`](../specs/evaluation-contract.md)
- Recommendation protocol: [`../specs/recommendation-protocol.md`](../specs/recommendation-protocol.md)
- Design: [`../superpowers/specs/2026-10-05-t5-1-evaluation-design.md`](../superpowers/specs/2026-10-05-t5-1-evaluation-design.md)
- Implementation plan: [`../superpowers/plans/2026-10-05-t5-1-evaluation.md`](../superpowers/plans/2026-10-05-t5-1-evaluation.md)

## Evaluation inputs

`src/trustrec/evaluation/protocol.py` builds one case for each user with a
new positive target at or after the selected cutoff. A candidate is a known
item before the cutoff that the user did not review before that cutoff.

The evaluator reads `train_interactions` and `validation_targets` for `t0`.
It reads `fit_interactions` and `test_targets` for `t1`. It rejects a source
row at or after the cutoff. It records target exclusions for old targets,
non-positive ratings, unknown items, history items, duplicate pairs, review
overlap, and empty candidate sets.

Each model input is a JSON Lines file with one row per eligible user:

```json
{
  "user_id": "u1",
  "snapshot_id": "video_games-full-d6c4efeb74aa",
  "cutoff_timestamp": "2020-08-31T22:46:09.247000+00:00",
  "candidate_item_ids": ["a", "b"],
  "ranked_item_ids": ["b", "a"],
  "explanation_coverage": 1.0,
  "latency_ms": 2.4
}
```

The runner requires the same users, candidate IDs, and target IDs for every
model. It records a candidate-set hash and a SHA-256 hash for every input.

## Reported results

The output reports NDCG, Recall, and Precision at 5, 10, and 20. It reports
coverage, candidate retention, aspect diversity, explanation coverage, and
latency. It records fit time and memory when ranking rows provide those values.

The output includes low, medium, and high item-popularity slices. Popularity
uses positive interaction counts before the cutoff. It includes history groups
`1-2`, `3-5`, and `>5`, plus a `no_history` count.

Each metric has a paired user bootstrap interval. The default uses 2,000
replicates, a 0.95 confidence level, and seed 7. Final random model runs use
seeds 7, 17, and 27 and record those seeds in their manifests.

## Reproducible command

Run the command from the repository root after the snapshot and ranking files
exist. Repeat it for each final model seed.

```bash
PYTHONPATH=src python3 scripts/run_evaluation.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --cutoff t1 \
  --model b0_most_popular=artifacts/rankings/b0_t1.jsonl \
  --model b1_item_knn=artifacts/rankings/b1_t1.jsonl \
  --model b2_bpr_mf=artifacts/rankings/b2_t1_seed07.jsonl \
  --model b3_ppr=artifacts/rankings/b3_t1.jsonl \
  --model h0_fixed_hybrid=artifacts/rankings/h0_t1.jsonl \
  --model t0_trustrec=artifacts/rankings/t0_t1.jsonl \
  --run-id t5-1-video-games-t1-seed07 \
  --output reports/generated/t5-1-video-games-t1-seed07.json \
  --manifest reports/generated/t5-1-video-games-t1-seed07.manifest.json \
  --reference-model b0_most_popular
```

The command writes a metrics artifact and a run manifest. The manifest stores
the snapshot ID, dataset hash, cutoff values, configuration hash, model IDs,
seed, source hashes, ranking hashes, output hash, and status. A completed
manifest cannot be replaced by another run with the same path.

## Validation

The fixture tests use fixed interactions, targets, IDs, timestamps, and seeds.
They cover metric formulas, paired bootstrap pairing, temporal exclusions,
candidate-set equality, artifact lineage, and the no-overwrite rule.

```bash
python3 -m pytest tests/unit/test_evaluation_metrics.py \
  tests/unit/test_evaluation_protocol.py \
  tests/integration/test_evaluation_script.py -q
```

The full-category metric tables remain external because generated Parquet and
model result files do not belong in Git.
