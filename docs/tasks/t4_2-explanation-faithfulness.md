# T4.2: Explanation and faithfulness checks

Status: done.

This task selects evidence-backed claims for recommendations. It refuses a
strong claim when support is below the configured thresholds. It records
mixed positive and negative evidence. Each selected passage stores its review
ID and character offsets.

## Related records

- Backlog: [`backlog.md`](backlog.md)
- Explanation contract: [`../specs/evidence-and-explanation.md`](../specs/evidence-and-explanation.md)
- Audit contract: [`../specs/explanation-audit-contract.md`](../specs/explanation-audit-contract.md)
- Abstention decision: [`../adr/0010-explanation-abstention.md`](../adr/0010-explanation-abstention.md)
- Lineage decision: [`../adr/0011-experiment-and-claim-lineage.md`](../adr/0011-experiment-and-claim-lineage.md)
- Phase plan: [`../plans/phase-4-trustrec-explanations.md`](../plans/phase-4-trustrec-explanations.md)

## Source and cutoff

The production source snapshot is `video_games-full-d6c4efeb74aa`. Its dataset
hash is
`0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`.
The training cutoff is `2019-01-13T04:24:39.377000+00:00`.
The audit rejects evidence at or after the supplied cutoff.

## Claim selection

`src/trustrec/explanations/claims.py` ranks aspects by user weight multiplied
by the item score deviation from its prior. It emits the highest ranked
claims up to `max_claims`.

Each strong claim contains the item, aspect, sentiment direction, support,
author count, effective sample size, and exact evidence text. A mixed claim
keeps positive and negative passages. A refused claim keeps its refusal
reason and does not emit strong claim text.

The explanation status is `supported`, `insufficient_support`, or
`unavailable`. Score parts are kept with the explanation. The dominant score
part is recorded so an aspect claim does not hide a larger MF or graph part.

## Faithfulness audit

`src/trustrec/explanations/faithfulness.py` removes all reviews cited by a
claim. It recomputes the local aspect contribution and keeps every other
score part fixed. It also removes a deterministic random set of the same
size. The result stores both recomputed contribution maps and both score
changes.

An audit result is `faithful` when the cited removal changes the score more
than the random removal and exceeds the configured minimum effect. It is
`not_faithful` when this test fails. Refused claims produce `abstained` audit
rows. The audit stores the candidate set hash when candidate IDs are supplied
and always records that normalization stayed fixed.

## Artifact and command

The runner is `scripts/run_explanation_audit.py`. It writes a JSON artifact
that includes the snapshot ID, dataset hash, cutoff, source hashes,
configuration hash, model hash, row counts, audit rows, and removal details.
The schema is [`../../schemas/explanation-audit.schema.json`](../../schemas/explanation-audit.schema.json).

Run the deterministic smoke audit from the repository root:

```bash
PYTHONPATH=src python3 scripts/run_explanation_audit.py \
  --evidence tests/fixtures/t4_2_evidence.jsonl \
  --claims tests/fixtures/t4_2_claims.json \
  --snapshot-id video_games-full-d6c4efeb74aa \
  --dataset-hash 0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043 \
  --cutoff-timestamp 2019-01-13T04:24:39.377000+00:00 \
  --output /tmp/trustrec-t4-2-audit.json
```

The smoke fixture is synthetic. It checks the artifact path and lineage
fields. Recommendation evaluation uses the real temporal interaction test.

## Reproduce the checks

```bash
ruff format --check src/trustrec/explanations scripts/run_explanation_audit.py src/trustrec/schemas/contracts.py tests/unit/test_explanation_claims.py tests/unit/test_faithfulness.py tests/integration/test_explanation_audit_script.py
ruff check src/trustrec/explanations scripts/run_explanation_audit.py src/trustrec/schemas/contracts.py tests/unit/test_explanation_claims.py tests/unit/test_faithfulness.py tests/integration/test_explanation_audit_script.py
python3 -m pytest tests/unit/test_explanation_claims.py tests/unit/test_faithfulness.py tests/integration/test_explanation_audit_script.py -q
```

The tests use fixed rows and seeds. They do not download source data.
