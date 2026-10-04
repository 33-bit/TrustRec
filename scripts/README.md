# Scripts

Scripts are reproducible entry points. They must accept clear input, configuration, and output paths. They must print a short run summary. They must preserve source hashes. They must stop on schema or leakage errors. Do not use personal paths or an implicit “latest” artifact.

Use `make snapshot-full` to scan the full Video Games review and metadata files. The command writes a benchmark manifest and Parquet tables under `data/`. It streams records in batches and does not store raw source files.

Use `scripts/build_llm_pseudo_test.py` to rebuild the held-out source selection. Pass the full snapshot manifest, pilot JSONL path, and output path. Use `scripts/validate_llm_pseudo_test.py` to make sure that the frozen records match their manifest and source snapshot.
