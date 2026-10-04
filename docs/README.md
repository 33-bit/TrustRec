# TrustRec Documentation Map

This directory stores decisions, contracts, plans, and run records. The supplied design is kept at [`design/trustrec-design-source.md`](design/trustrec-design-source.md) as a fixed reference. Later decisions are in ADR-0013 and the active specs.

## Read in this order

1. [`design/trustrec-design-source.md`](design/trustrec-design-source.md): the full research and system design.
2. [`adr/`](adr/): decisions that guide implementation. ADR-0013 defines the LLM-only annotation policy.
3. [`specs/`](specs/): interfaces, evaluation rules, and the explanation audit.
4. [`plans/`](plans/): work order and exit rules.
5. [`tasks/backlog.md`](tasks/backlog.md): task status and dependencies.
6. [`reference/project-status.md`](reference/project-status.md): measured, pilot, and pending work.

## Artifact policy

`docs/tasks/` contains small decision artifacts and LLM annotation records. Keep large data, model weights, Parquet snapshots, caches, and generated figures outside Git. Each experiment must name a dataset hash, snapshot ID, model or prompt version, seed, and protocol version.
