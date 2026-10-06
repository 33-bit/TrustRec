# T5.3 Report and Reproducibility Bundle Design

## Goal

Package measured evidence and explicit limits into a report that can be traced to immutable source files and commands.

## Scope

T5.3 owns report assembly, claim lineage, source hashes, command records, and bundle verification.
T5.1 owns recommendation metric generation.
T5.2 owns the Streamlit demo and its synthetic fallback.
The package does not download data or execute a ranking run.

## Input contract

`configs/report_bundle.json` names source artifacts, commands, runs, and claims.
Each source path is relative to the repository root.
Required files must exist and match their SHA-256 value.
External files can be absent, but the report records the path and unavailable hash state.

Each run points to a manifest artifact and a command.
Available runs require complete status, matching run and snapshot fields, and matching metric hashes.
Recommendation metrics require the candidate rule, target rule, candidate-set hash, and one eligible-user count across models.
LLM pseudo-test metrics require the frozen split and disabled tuning.

Each claim has one kind, one run, and one or more JSON pointers.
The packager resolves scalar values from source artifacts.
Measured claims cannot cite synthetic or design runs.
Unavailable external runs can support only limitation claims.
The rendered claim includes links to its run manifest and command.

## Outputs

The packager writes a Markdown report, a CSV result table, a resolved claim file, a command list, the normalized specification, a bundle manifest, and a checksum inventory.
It copies only listed environment files and source manifests.
It writes to a temporary sibling directory and renames that directory after all files pass validation.
An existing output directory is never replaced.

The verifier checks generated file hashes, copied source hashes, and every checksum line.
The bundle contains no raw review text, Parquet tables, model weights, or complete metric records.

## Default evidence

The default specification uses the full snapshot manifest, the T2.3 NLP metrics and manifest, and a small T5.2 demo manifest.
It includes two external T5.1 paths so the report can state that recommendation values are unavailable.
The report contains measured snapshot and NLP claims, a synthetic demo claim, and a recommendation limitation.

## Testing

Unit tests use small JSON fixtures and fixed hashes.
Integration tests call the real CLI.
Contract tests inspect the machine-readable schema.
No test downloads production data.
