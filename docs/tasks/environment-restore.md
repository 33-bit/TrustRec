# Restore the working environment in a new session

The cloud container is temporary. Code and records live in Git. Data, models,
evidence, and the LLM cache live in ignored folders and are lost with the
container. Run these steps from the repository root in a new session.

1. Install the locked dependencies.

   ```bash
   make install
   ```

2. Rebuild the snapshot. This takes about 7 minutes and needs network access to `huggingface.co`.

   ```bash
   make snapshot-full
   ```

   Six of seven tables match the recorded manifest byte for byte. `item_metadata` can differ by platform (see T6.4).

3. Rebuild dictionary evidence. This takes about 11 minutes per cutoff.

   ```bash
   PYTHONPATH=src python3 scripts/label_corpus_evidence.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cutoff t0 --workers 4 --output artifacts/evidence/t0_dictionary.jsonl
   PYTHONPATH=src python3 scripts/label_corpus_evidence.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cutoff t1 --workers 4 --output artifacts/evidence/t1_dictionary.jsonl
   ```

4. Restore the LLM cache. Copy the downloaded `deepseek_t1_labels.jsonl` and `deepseek_t1_labels.runs.jsonl` into `artifacts/llm/`. Then resume labeling. The command reads `DEEPSEEK_API` and `MODEL` from the environment and skips cached reviews.

   ```bash
   PYTHONPATH=src python3 scripts/label_reviews_llm.py --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json --cache artifacts/llm/deepseek_t1_labels.jsonl --workers 32 --batch-size 20 --min-balance 0.5
   ```

   Pass `--model <name>` to use a model other than `MODEL`. A different model or reasoning setting labels every review again and never mixes with cached labels.

5. Rebuild rankings and models with the commands in [T6.4](t6_4-baseline-results.md). Seeds make every run reproducible.
