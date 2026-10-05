"""Validate the repository's foundational files and directory contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "Makefile",
    ".python-version",
    "requirements.lock",
    "configs/protocol.toml",
    "docs/superpowers/specs/2026-10-02-trustrec-foundation-design.md",
    "docs/superpowers/plans/2026-10-02-trustrec-foundation.md",
    "docs/README.md",
    "docs/reference/project-status.md",
    "docs/tasks/t0_2-environment.md",
    "docs/tasks/t0_2-install.log",
    "docs/tasks/t1_2_snapshot_manifest.json",
    "docs/specs/evaluation-contract.md",
    "docs/specs/nlp-annotation-contract.md",
    "docs/specs/explanation-audit-contract.md",
    "docs/tasks/t4_2-explanation-faithfulness.md",
    "docs/tasks/t5_1-evaluation.md",
    "docs/adr/0013-llm-only-annotation-policy.md",
    "docs/tasks/llm-annotation-split-manifest.template.json",
    "docs/tasks/t2_2-llm-pseudo-test.md",
    "docs/tasks/t2_2_llm_pseudo_test.manifest.json",
    "scripts/build_full_snapshot.py",
    "scripts/build_llm_pseudo_test.py",
    "scripts/validate_llm_pseudo_test.py",
    "schemas/trustrec-artifacts.schema.json",
    "schemas/annotation-record.schema.json",
    "schemas/explanation-audit.schema.json",
    "scripts/run_explanation_audit.py",
    "scripts/run_evaluation.py",
    "scripts/package_report.py",
    "configs/report_bundle.json",
    "schemas/report-bundle.schema.json",
    "docs/specs/report-bundle-contract.md",
    "docs/superpowers/specs/2026-10-06-t5-3-report-bundle-design.md",
    "docs/plans/t5_3-report-bundle.md",
    "docs/tasks/t5_3-report-bundle.md",
    "app/demo_bundle.manifest.json",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required repository files:\n- " + "\n- ".join(missing))
    print(f"Validated {len(REQUIRED_FILES)} repository foundation files")


if __name__ == "__main__":
    main()
