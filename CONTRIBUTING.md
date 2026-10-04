# Contributing to TrustRec

1. Read `AGENTS.md`, the related ADR, and the related spec before you change a contract.
2. Add or update a task in `docs/tasks/backlog.md`. State the acceptance checks.
3. Record snapshot IDs, cutoff timestamps, seeds, dataset hashes, and model hashes.
4. Add deterministic tests for new behavior. Use fixtures instead of production data.
5. Run `make validate` and `make check` before you open a pull request.

Use Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `test:`, and `chore:`. A pull request must explain the change, link its task and spec, list validation results, and include dashboard screenshots when the UI changes.
