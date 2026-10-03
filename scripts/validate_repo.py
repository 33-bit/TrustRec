"""Validate the repository's foundational files and directory contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "Makefile",
    "configs/protocol.toml",
    "docs/superpowers/specs/2026-10-02-trustrec-foundation-design.md",
    "docs/superpowers/plans/2026-10-02-trustrec-foundation.md",
    "docs/README.md",
    "docs/reference/project-status.md",
    "docs/specs/evaluation-contract.md",
    "docs/specs/nlp-annotation-contract.md",
    "schemas/trustrec-artifacts.schema.json",
    "schemas/annotation-record.schema.json",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required repository files:\n- " + "\n- ".join(missing))
    print(f"Validated {len(REQUIRED_FILES)} repository foundation files")


if __name__ == "__main__":
    main()
