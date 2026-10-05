# Report and reproducibility bundle contract

T5.3 packages existing evidence into a report and a reproducibility index.
A manifest records the inputs and outputs of a run.
The packager reads files and never runs the commands that the specification lists.

## Inputs

Use `scripts/package_report.py --spec configs/report_bundle.json --output-dir reports/generated/t5-3` from the repository root.
The JSON specification contains `schema_version`, `bundle_id`, `title`, `environment_files`, `artifacts`, `commands`, `runs`, and `claims`.
Identifiers contain letters, numbers, periods, underscores, or hyphens.
All artifact and environment paths are relative to the repository root.
Reject absolute paths, parent traversal, and symbolic links that leave that root.

Each artifact names its `path`, expected `sha256`, and `availability`.
Availability is `required` or `external`.
Every required artifact must exist and match its hash.
If an external artifact exists, it must also match its hash.
If an external artifact is absent, record its expected hash when one is known.
If no hash is available, record the unavailable hash state.
An absent artifact cannot supply a claim value or a run manifest.

Each command names a `command_id`, command text, and prerequisite text.
Commands run from the repository root.
Record dependencies and new output paths before commands that produce another run.
Never replace a completed experiment to reproduce an old result.

## Runs and claims

Each run names a unique `run_id`, `manifest_artifact`, `command_id`, and `evidence_role`.
It can also name a `metrics_artifact`.
Evidence roles are `snapshot`, `recommendation_test`, `llm_pseudo_test`, `synthetic`, and `design`.
A run ID must match its manifest.
A snapshot uses its `snapshot_id` as the run ID when its manifest has no run ID.
Other runs require `status: complete`.

The packager records snapshot ID, dataset hash, cutoff, protocol, seeds, configuration hash, model hashes, and code revision.
Record unavailable source fields as `unavailable`.
Keep declared model seeds separate from evidence of executed runs.
A list of seeds in one manifest does not prove that three runs occurred.

If a run supplies metrics, make sure that their hash matches `artifact_sha256.metrics` in its manifest.
Make sure that their run ID, snapshot ID, dataset hash, cutoffs, protocol, and seeds agree with the manifest.
For NLP results, require the frozen pseudo-test and the prohibition on tuning.
For recommendation results, require the locked candidate hash, candidate rule, target rule, eligible-user count, and bootstrap description.
Model records must use the same candidate hash and eligible-user count.

Each claim contains `claim_id`, `kind`, `statement`, `run_id`, and a nonempty `evidence` list.
Claim kinds are `measured`, `design`, `limitation`, and `synthetic`.
Each evidence entry contains an artifact ID, a label, and a JSON pointer.
A JSON pointer identifies one value inside an artifact.
The packager resolves pointers and records scalar values without copying whole source records.
Claims can cite their run manifest or metrics artifact.
Every rendered claim links to its manifest and reproduction command.

Synthetic evidence supports only synthetic claims and limitations.
Design evidence supports only design claims and limitations.
Recommendation metrics and LLM consistency results appear in separate sections.
The report describes LLM results as consistency with pseudo-labels.
The report does not infer human agreement, model superiority, or missing experiment results.

## Outputs

The output directory contains `report.md`, `claims.json`, `results.csv`, `commands.md`, `specification.json`, and `bundle.manifest.json`.
It also contains copies of the listed environment files and source manifests.
The bundle never copies raw reviews, model weights, ranking files, or complete metrics records.
The manifest records hashes for every source and generated file.
`SHA256SUMS` includes the bundle manifest and the other generated files.

The result table includes claim ID, evidence role, run ID, model ID, snapshot ID, cutoff, seed, metric label, and value.
Recommendation tables also include eligible users, candidate hash, units, and bootstrap intervals.
The report preserves resources, slices, exclusion counts, and declared seeds from supplied recommendation metrics.
Unavailable measurements stay unavailable.

Use `scripts/package_report.py --verify-bundle PATH` to make sure that a saved bundle matches its hashes.
The verifier uses only the bundle files and does not require production data.
If the output directory already exists, stop without changing it.
Validate all inputs before creating output files.
Publish the output directory only after every file is ready.
Identical inputs and code revision produce identical bytes in different output directories.

## Evidence boundary

The default specification packages the full snapshot manifest, recorded NLP consistency metrics, and the synthetic T5.2 fallback.
The package states that full recommendation metrics are absent.
Missing Parquet tables remain external dependencies with their recorded hashes.
This package completes the report workflow and does not complete an unrun full recommendation experiment.

Related rules are [result reporting](result-reporting.md), [experiment manifests](experiment-manifest.md), and [ADR-0011](../adr/0011-experiment-and-claim-lineage.md).
