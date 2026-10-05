# T4.1: Evidence aggregation and adaptive gate

Status: done.

This task adds snapshot-safe evidence profiles and two hybrid rankers. H0 is
the fixed-weight hybrid. T0 is the adaptive TrustRec gate. Both rankers use
the same percentile normalization for MF, graph, and aspect scores.

## Related records

- Backlog: [`backlog.md`](backlog.md)
- Evidence contract: [`../specs/evidence-aggregation-contract.md`](../specs/evidence-aggregation-contract.md)
- Ranking contract: [`../specs/ranking-contract.md`](../specs/ranking-contract.md)
- Score decision: [`../adr/0009-score-normalization-and-gating.md`](../adr/0009-score-normalization-and-gating.md)
- Data contract: [`../specs/data-contract.md`](../specs/data-contract.md)
- Phase plan: [`../plans/phase-4-trustrec-explanations.md`](../plans/phase-4-trustrec-explanations.md)

## Source and cutoff

The source snapshot is `video_games-full-d6c4efeb74aa`. Its dataset hash is
`0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043`.
The training cutoff is `2019-01-13T04:24:39.377000+00:00`.
The final fit cutoff is `2020-08-31T22:46:09.247000+00:00`.

The aggregator rejects evidence at or after its supplied cutoff. It accepts
flat evidence rows and annotation records with nested labels. A review and
aspect pair contributes once after clause aggregation.

## Evidence profiles

`src/trustrec/explanations/evidence.py` implements the review weight

```text
q = confidence × duplicate_penalty × recency
```

It applies inverse duplicate-group size by default. A non-zero recency half
life uses exponential decay. The profile stores shrunken score, support,
positive mass, negative mass, neutral mass, author count, review count, and
effective sample size. Missing evidence keeps the supplied prior and has zero
support. Conflicting positive and negative mass remains visible.

`compute_aspect_priors` uses only evidence before the supplied cutoff.
`build_user_aspect_weights` applies smoothed review counts and returns weights
that sum to one.

Use `scripts/build_evidence_profiles.py` to write a JSON profile artifact. The
artifact records the source evidence hash, snapshot ID, dataset hash, cutoff,
configuration hash, and model hash.

## Hybrid rankers

`src/trustrec/recommenders/hybrid.py` contains the shared percentile
normalization. Average ranks resolve ties. An all-equal component receives
`0.5` for every candidate.

H0 uses the configured weights from
[`h0_fixed_hybrid.toml`](../../configs/experiments/h0_fixed_hybrid.toml).
T0 uses

```text
g = g_max × kappa / (kappa + history_count) × support
base = rho × normalized_mf + (1 - rho) × normalized_graph
score = (1 - g) × base + g × normalized_aspect
```

The functions reject mismatched candidate IDs. The details API stores the
normalized parts, contributions, gate, support, and history count.

Use `scripts/run_hybrid.py` with prepared MF, graph, and aspect artifacts.
The output records common candidate IDs, source model hashes, configuration
hash, model hash, and score parts. H0 and T0 use the same input artifacts and
the same normalization path, so their validation budget can stay equal.

## Reproduce the checks

Run these commands from the repository root:

```bash
ruff format --check src/trustrec/explanations src/trustrec/recommenders scripts/run_hybrid.py scripts/build_evidence_profiles.py tests/unit/test_evidence_aggregation.py tests/unit/test_hybrid.py tests/integration/test_hybrid_script.py tests/integration/test_evidence_profiles_script.py
ruff check src/trustrec/explanations src/trustrec/recommenders scripts/run_hybrid.py scripts/build_evidence_profiles.py tests/unit/test_evidence_aggregation.py tests/unit/test_hybrid.py tests/integration/test_hybrid_script.py tests/integration/test_evidence_profiles_script.py
python3 -m pytest tests/unit/test_evidence_aggregation.py tests/unit/test_hybrid.py tests/integration/test_hybrid_script.py tests/integration/test_evidence_profiles_script.py -q
```

The tests use fixed rows and do not download source data. The full repository
suite passed with 135 tests.
