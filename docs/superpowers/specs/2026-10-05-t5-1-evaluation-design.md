# T5.1 Evaluation Design

## Goal

T5.1 produces one reproducible evaluation artifact for baseline, ablation,
and sparse-history comparisons. The artifact contains metric values, paired
user bootstrap intervals, slice counts, exclusions, and lineage fields.

The evaluator uses the locked temporal recommendation protocol. It does not
use the frozen LLM pseudo-test for recommendation metrics.

## Scope

The change adds a reusable evaluation module and one command line runner.
The module accepts prepared per-user ranking results. This keeps model fitting
separate from metric calculation and lets the same candidate and target sets
serve every model. The runner reads snapshot tables and ranking artifacts,
checks their lineage, and writes a metrics artifact beside a run manifest.

The change does not download source data, generate LLM labels, or change any
recommender implementation. Full-category results remain external because
the repository does not store generated Parquet tables or model outputs.

## Data flow

1. Read the snapshot manifest and the interaction table named by the selected
   cutoff. Read the validation or recommendation test target table.
2. Build one immutable evaluation case for each target user. A case stores the
   user ID, candidate IDs, target IDs, history count, item popularity groups,
   and target exclusion reasons.
3. Load one ranking artifact per model. Each row stores the user ID, candidate
   IDs, ranked item IDs, optional explanation status, and model lineage.
4. Require every model to declare the same users, candidate sets, and target
   sets. Reject unknown items, duplicate ranks, history items, post-cutoff
   rows, and incomplete ranking artifacts.
5. Compute per-user metrics, aggregate metrics, slice counts, and resource
   summaries. Compute paired user bootstrap intervals from the same sampled
   user indices for every compared model.
6. Write a metrics JSON artifact and a manifest. A completed manifest cannot
   be replaced by another run with the same run ID.

## Interfaces

`src/trustrec/evaluation/metrics.py` owns pure metric functions and typed
records. It exposes `ndcg_at_k`, `recall_at_k`, `precision_at_k`,
`intra_list_diversity`, and `paired_user_bootstrap`.

`src/trustrec/evaluation/protocol.py` owns evaluation cases, temporal checks,
slice assignment, per-user aggregation, and model-set equality checks. It
exposes `build_evaluation_cases`, `evaluate_model_rankings`, and
`compare_model_rankings`.

`scripts/run_evaluation.py` reads Parquet and JSON Lines inputs, calls the
protocol module, and writes the result artifact. Ranking rows use this shape:

```json
{
  "user_id": "u1",
  "snapshot_id": "snap-1",
  "cutoff_timestamp": "2020-01-01T00:00:00+00:00",
  "candidate_item_ids": ["a", "b"],
  "ranked_item_ids": ["b", "a"],
  "explanation_coverage": 1.0,
  "latency_ms": 2.4
}
```

The runner accepts `MODEL_ID=PATH` pairs. Each path must contain one row for
each eligible user. The runner accepts optional item aspect sets for diversity
and optional explanation coverage values. It records missing optional values
as unavailable and does not treat them as zero.

## Metrics

The evaluator reports NDCG, Recall, and Precision at K values 5, 10, and 20.
It reports recommendation coverage, intra-list diversity when item aspects
exist, explanation coverage when explanation values exist, and mean latency.
It also reports eligible users, evaluated users, candidate retention, and
explicit exclusion counts.

Popularity slices use the item positive interaction count before the cutoff.
They use low, medium, and high groups with deterministic rank boundaries.
Sparse-history slices use the configured history groups `1-2`, `3-5`, and
`>5`. Users with no history are excluded from sparse-history metrics and are
counted under `no_history`.

Metric values are macro averages over eligible users. A target item counts as
relevant once. A ranked list can contain at most one occurrence of each item.
Coverage is the fraction of eligible users with at least one ranked item.
Diversity is one minus the mean pairwise Jaccard similarity of item aspect
sets in the evaluated list. A one-item list has diversity zero.

## Bootstrap and seeds

`paired_user_bootstrap` resamples users with replacement using one random
index array per replicate. It returns the point estimate, lower and upper
confidence limits, sample count, confidence level, and seed. Model deltas use
the same resampled user indices, so paired uncertainty remains aligned.

The default configuration uses seeds 7, 17, and 27 and 2,000 bootstrap
replicates. Deterministic models can use one seed. Random models must provide
all three seeds in their run manifests. Bootstrap uses the locked user set and
never reads target rows while fitting a model.

## Ablations

The runner accepts the model IDs in `configs/experiments/ablations.toml`.
Each ablation must point to a ranking artifact built from the same snapshot,
cutoff, users, candidates, targets, and tuning budget as its comparison
models. The output records the comparison group and the source model hash for
each artifact.

## Failure behavior

The runner stops with a clear error when a required field is missing, a path
does not exist, a JSON Lines row is malformed, a model uses a different
candidate or target set, or a timestamp reaches the cutoff. It writes a
failed manifest with the error and environment details when the run has a
valid run ID. It never overwrites a completed manifest.

## Validation

Unit tests cover each metric, tie handling, empty lists, sparse and popularity
slices, bootstrap pairing, and invalid rows. An integration test uses small
Parquet and JSON Lines fixtures to verify lineage, output hashes, set
equality, and the no-overwrite rule. The full repository test suite remains
the final gate.
