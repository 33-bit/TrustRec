from pathlib import Path


def test_required_directory_contract_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "src/trustrec",
        "tests",
        "configs",
        "configs/experiments",
        "docs/adr",
        "docs/design",
        "docs/specs",
        "docs/plans",
        "docs/tasks",
        "schemas",
        "reports",
    ):
        assert (root / relative).is_dir(), relative
