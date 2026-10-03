# Contributing to TrustRec

1. Read `AGENTS.md`, the relevant ADR, and the spec before changing a contract.
2. Create or update a task in `docs/tasks/backlog.md` and state its acceptance checks.
3. Keep snapshot IDs, cutoff timestamps, seeds, dataset hashes, and model hashes in experiment outputs.
4. Add deterministic tests for new behavior. Unit tests must use fixtures, not downloaded production data.
5. Run `make validate` and `make check` before opening a pull request.

Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`). A pull request should explain the change, link its task/spec, list validation commands and results, and include dashboard screenshots when UI behavior changes.

