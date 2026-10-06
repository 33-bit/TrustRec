"""Package and verify the T5.3 report and reproducibility bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustrec.reporting.bundle import git_revision, package_report, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=Path)
    group.add_argument("--verify-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--code-revision")
    args = parser.parse_args()
    try:
        if args.verify_bundle is not None:
            manifest = verify_bundle(args.verify_bundle)
            print(json.dumps({"status": "verified", "bundle_id": manifest["bundle_id"]}))
            return 0
        if args.output_dir is None:
            parser.error("--output-dir is required with --spec")
        manifest = package_report(
            args.spec,
            args.output_dir,
            repository_root=args.root,
            code_revision=args.code_revision or git_revision(args.root),
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "bundle_id": manifest["bundle_id"],
                    "output_dir": str(args.output_dir),
                    "claim_count": len(manifest["claims"]),
                }
            )
        )
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"package_report: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
