# T5.3: Report and reproducibility bundle

Status: done for the report packager and the recorded repository evidence.
The full recommendation result remains external until T5.1 ranking files exist.

## Related records

- Backlog: [`backlog.md`](backlog.md)
- Task matrix: [`task-matrix.md`](task-matrix.md)
- Result reporting contract: [`../specs/result-reporting.md`](../specs/result-reporting.md)
- Experiment manifest contract: [`../specs/experiment-manifest.md`](../specs/experiment-manifest.md)
- Bundle contract: [`../specs/report-bundle-contract.md`](../specs/report-bundle-contract.md)
- Design: [`../superpowers/specs/2026-10-06-t5-3-report-bundle-design.md`](../superpowers/specs/2026-10-06-t5-3-report-bundle-design.md)
- Implementation plan: [`../plans/t5_3-report-bundle.md`](../plans/t5_3-report-bundle.md)
- Lineage decision: [`../adr/0011-experiment-and-claim-lineage.md`](../adr/0011-experiment-and-claim-lineage.md)

## Outputs

The input specification is [`configs/report_bundle.json`](../../configs/report_bundle.json).
The command is [`scripts/package_report.py`](../../scripts/package_report.py).
The source library is [`src/trustrec/reporting/bundle.py`](../../src/trustrec/reporting/bundle.py).
The machine-readable contract is [`schemas/report-bundle.schema.json`](../../schemas/report-bundle.schema.json).
The generated bundle belongs in ignored `reports/generated/t5-3/`.
The checked-in bundle record is [`t5_3_reproducibility.manifest.json`](t5_3_reproducibility.manifest.json).
The bundle contains `report.md`, `claims.json`, `results.csv`, `commands.md`, `specification.json`, `bundle.manifest.json`, `SHA256SUMS`, and small source copies.

Run the command from the repository root:

```bash
PYTHONPATH=src python3 scripts/package_report.py \
  --spec configs/report_bundle.json \
  --output-dir reports/generated/t5-3
PYTHONPATH=src python3 scripts/package_report.py \
  --verify-bundle reports/generated/t5-3
```

The command validates every required artifact before it creates output.
It rejects path traversal, hash changes, duplicate IDs, mismatched run fields, and claims without a matching run and command.
It does not run the listed experiment commands.
It does not copy raw reviews, Parquet tables, model weights, ranking files, or complete metrics records.

## Recorded evidence

The package uses snapshot `video_games-full-d6c4efeb74aa`.
Its dataset hash is `0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`.
The snapshot contains 1,717,097 interaction rows and 49,749 unique items.
The snapshot cutoff values are `2019-01-13T04:24:39.377000+00:00` and `2020-08-31T22:46:09.247000+00:00`.

The NLP evidence uses run `t2.3-nlp-baselines-v1`, seed 7, protocol `nlp-baseline-v1`, and configuration hash `3d02bd77ef0f48d910b15934877e4585166d00e2f3d1f52c66fe8844f2f8cee0`.
It measures consistency with the frozen LLM pseudo-test.
The dictionary aspect micro F1 is 0.5085.
The TF-IDF plus Linear SVM aspect micro F1 is 0.0920.
The dictionary evidence offset validity rate is 1.0000.

The T5.2 fallback uses the synthetic bundle manifest [`app/demo_bundle.manifest.json`](../../app/demo_bundle.manifest.json).
It is a presenter fixture and does not support a production metric claim.

Every claim in the generated report links to its run manifest and reproduction command.
Each claim also names a JSON pointer and a source hash.
The result table keeps the model, snapshot, cutoff, seed, protocol, configuration hash, and uncertainty fields.

## Limits

The full recommendation metric artifact is not in this checkout.
The report marks its manifest and metrics path as external and unavailable.
The reproduction command names the full T5.1 command and its ranking inputs.
The report does not invent recommendation values.

The fallback bundle metric is synthetic.
The NLP values measure pseudo-label consistency.
They do not measure human agreement or ground-truth accuracy.

## Validation

The focused report tests cover valid packaging, deterministic bytes, hash failures, path traversal, duplicate IDs, run lineage, scalar pointers, external artifacts, the CLI, and saved bundle checksums.
The repository gate is `make check`.
The repository file gate is `make validate`.
