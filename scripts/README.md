# Scripts

Scripts are reproducible entry points. They must accept clear input, configuration, and output paths. They must print a short run summary. They must preserve source hashes. They must stop on schema or leakage errors. Do not use personal paths or an implicit “latest” artifact.

Use `make snapshot-full` to scan the full Video Games review and metadata files. The command writes a benchmark manifest and Parquet tables under `data/`. It streams records in batches and does not store raw source files.

Use `scripts/build_llm_pseudo_test.py` to rebuild the held-out source selection. Pass the full snapshot manifest, pilot JSONL path, and output path. Use `scripts/validate_llm_pseudo_test.py` to make sure that the frozen records match their manifest and source snapshot.

Use `scripts/run_ranking_baselines.py` to run B0 or B1 from a prepared
snapshot. The command reads the `train_interactions` or
`fit_interactions` artifact named by the cutoff and writes a JSON result
with snapshot, dataset, cutoff, configuration, source artifact, and model
hashes.

```bash
python3 scripts/run_ranking_baselines.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --model b0_most_popular \
  --user-id USER_ID \
  --cutoff t0 \
  --k 10 \
  --output /tmp/trustrec-ranking.json
```
