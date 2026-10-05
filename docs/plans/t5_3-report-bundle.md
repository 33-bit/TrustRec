# T5.3 Report Bundle Implementation Plan

Goal: Package a report whose claims link to hashed manifests and reproduction commands.

Architecture: A Python library reads the specification, enforces source lineage, and prepares deterministic outputs.
A CLI publishes the directory and verifies saved output hashes.
The default package combines existing measured evidence with labeled synthetic examples.

Tech stack: Python 3.11 or newer, standard library, pytest, and Ruff.

Spec: [`../specs/report-bundle-contract.md`](../specs/report-bundle-contract.md).

## Constraints

Keep raw data and generated model files outside Git.
Use fixed seeds in tests and do not download production data.
Keep completed artifacts unchanged.
Keep artifact paths relative to the repository root.
Store the task record in `docs/tasks/`.
Use the current checkout on branch `codex/t5-3-report-bundle`.
The user approved this design and implementation in the current chat.

## Task 1: Source lineage and deterministic packaging

Files: Create `src/trustrec/reporting/{__init__,bundle,render}.py`, `tests/conftest.py`, and `tests/unit/test_report_bundle.py`.

Interface: `package_report(spec_path: Path, output_dir: Path, *, repository_root: Path, code_revision: str) -> dict[str, Any]`.
The function returns the saved bundle manifest.

- [ ] Write tests for report links, extracted values, missing files, hash changes, duplicate IDs, pointers, path traversal, and completed output protection.
- [ ] Run `python3 -m pytest tests/unit/test_report_bundle.py -q` and record the expected failures before implementation.
- [ ] Implement input parsing, hashes, run agreement, evidence-role rules, scalar extraction, and atomic directory publication.
- [ ] Render the report, CSV values, commands, claims, source manifests, environment files, and checksum file.
- [ ] Run the unit tests and make sure that equivalent inputs produce identical output bytes.

The result test uses one declared model with an eligible-user count of two.
It supplies NDCG@10 `0.5` and interval `[0.25, 0.75]`.
Changing a referenced artifact after its hash is locked must stop packaging.

```python
result = package_report(spec_path, output_dir, repository_root=root, code_revision="fixture")
assert result["claims"][0]["evidence"][0]["value"] == 0.5
assert not (output_dir / "raw").exists()
```

## Task 2: CLI and saved bundle contract

Files: Create `scripts/package_report.py`, `schemas/report-bundle.schema.json`, `tests/integration/test_report_bundle_script.py`, and `tests/contract/test_report_bundle_schema.py`.

Interface: `verify_bundle(bundle_dir: Path) -> dict[str, Any]` rejects changed or missing output files.
The CLI accepts `--spec`, `--output-dir`, `--root`, and `--verify-bundle`.

- [ ] Write tests that run the real CLI, detect a modified report, and reject an incomplete bundle against the schema.
- [ ] Run the tests before implementation and record the expected failures.
- [ ] Add the CLI and schema without new runtime dependencies.
- [ ] Run the focused unit, integration, and contract tests.

```bash
PYTHONPATH=src python3 scripts/package_report.py --spec configs/report_bundle.json --output-dir reports/generated/t5-3
PYTHONPATH=src python3 scripts/package_report.py --verify-bundle reports/generated/t5-3
```

## Task 3: Concrete report, reproduction records, and task completion

Files: Create `configs/report_bundle.json`, `app/demo_bundle.manifest.json`, and `docs/tasks/t5_3-report-bundle.md`.
Update repository indexes, report instructions, and `scripts/validate_repo.py`.

- [ ] Build the default specification from the existing snapshot, NLP manifest, metrics, and demo hashes.
- [ ] Add exact reproduction commands and external dependencies.
- [ ] Generate and inspect the report and reproducibility bundle.
- [ ] Save a small bundle manifest in `docs/tasks/t5_3_reproducibility.manifest.json`.
- [ ] Make sure that report links, claim values, external file states, and synthetic labels match their sources.
- [ ] Run `make check`, `make validate`, and the saved bundle verifier.
- [ ] Review the implementation against the contract and fix findings.
- [ ] Record commands, results, output paths, and limits in the task record.
- [ ] Mark T5.3 done in the backlog and project status.

The full recommendation results remain absent until their external temporal tables and ranking files exist.
The task report must name this limit.
