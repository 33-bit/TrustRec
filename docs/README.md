# TrustRec Documentation Map

This directory separates decisions, stable contracts, execution plans, and run artifacts. The source design is copied to [`design/trustrec-design-source.md`](design/trustrec-design-source.md) so the repository has a fixed reference.

## Read in this order

1. [`design/trustrec-design-source.md`](design/trustrec-design-source.md) — full research/system design.
2. [`adr/`](adr/) — decisions that constrain implementation.
3. [`specs/`](specs/) — interfaces and evaluation rules that code must satisfy.
4. [`plans/`](plans/) — phase-level execution order and exit criteria.
5. [`tasks/backlog.md`](tasks/backlog.md) — current work status and dependencies.
6. [`reference/project-status.md`](reference/project-status.md) — what is measured, pilot-only, or still pending.

## Artifact policy

`docs/tasks/` contains small, reviewable decision artifacts and annotation worksheets. Large data, model weights, Parquet snapshots, caches, and generated figures belong outside Git under the paths described in `data/README.md` and `reports/README.md`. Every experiment must reference a dataset hash, snapshot ID, model/prompt version, seed, and protocol version.
