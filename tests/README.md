# Test Layout

- `tests/unit/`: pure functions and adapter contracts.
- `tests/contract/`: repository, schema, and manifest contracts.
- `tests/integration/`: snapshot/module integration using fixtures only.
- `tests/fixtures/`: small synthetic rows and expected outputs.

No test can download production data. Use fixed timestamps, IDs, and seeds for leakage and ranking fixtures.

