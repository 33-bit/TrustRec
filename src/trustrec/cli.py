"""Small command-line entry point used by the repository harness."""

from __future__ import annotations

import argparse

from trustrec import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustRec research pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
