# Scripts

Scripts are reproducible entry points. They must accept clear input, configuration, and output paths. They must print a short run summary. They must preserve source hashes. They must stop on schema or leakage errors. Do not use personal paths or an implicit “latest” artifact.

Use `make snapshot-full` to scan the full Video Games review and metadata files. The command writes a benchmark manifest and Parquet tables under `data/`. It streams records in batches and does not store raw source files.

Use `scripts/build_llm_pseudo_test.py` to rebuild the held-out source selection. Pass the full snapshot manifest, pilot JSONL path, and output path. Use `scripts/validate_llm_pseudo_test.py` to make sure that the frozen records match their manifest and source snapshot.

Use `scripts/run_ranking_baselines.py` to run B0 through B3 from a prepared
snapshot. The command reads the `train_interactions` or
`fit_interactions` artifact named by the cutoff and writes a JSON result
with snapshot, dataset, cutoff, configuration, source artifact, and model
hashes.

The command also runs B2 BPR MF and B3 personalized PageRank. B2 uses seed
7 when no seed is supplied. B3 uses damping 0.85 when no damping is supplied.

Use `scripts/build_evidence_profiles.py` to aggregate flat evidence JSON Lines
before a cutoff. The output keeps source, snapshot, configuration, and model
hashes. Use `scripts/run_hybrid.py` to combine prepared MF, graph, and aspect
scores for `h0_fixed_hybrid` or `t0_trustrec`. Both models require the same
candidate IDs and use the shared percentile normalizer.

Use `scripts/run_explanation_audit.py` to audit prepared claims. The command
removes cited reviews, compares the score change with a same-size random
removal, and writes snapshot and hash fields with every audit row.

Use `scripts/run_evaluation.py` to compare prepared ranking JSON Lines files.
The command reads the temporal snapshot tables, builds one shared user set,
computes ranking metrics and paired bootstrap intervals, and writes a metrics
artifact beside a run manifest. Pass one `--model MODEL_ID=PATH` argument for
each baseline or ablation.

```bash
PYTHONPATH=src python3 scripts/run_evaluation.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --cutoff t1 \
  --model b0_most_popular=artifacts/rankings/b0_t1.jsonl \
  --model t0_trustrec=artifacts/rankings/t0_t1.jsonl \
  --run-id t5-1-video-games-t1-seed07 \
  --output reports/generated/t5-1-video-games-t1-seed07.json
```

```bash
python3 scripts/run_ranking_baselines.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --model b0_most_popular \
  --user-id USER_ID \
  --cutoff t0 \
  --k 10 \
  --output /tmp/trustrec-ranking.json
```
