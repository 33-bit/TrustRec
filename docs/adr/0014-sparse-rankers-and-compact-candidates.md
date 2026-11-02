# ADR-0014: Sparse matrix rankers and compact candidate declarations

Status: Accepted. Date: 2026-11-02.

## Context

The T3 baselines and the T5.1 evaluator were correct on fixtures but could not
run on the full Video Games snapshot (1.5 million fit interactions, about one
million users, and about 50,000 items):

- BPR MF built a list of unknown items for every user before training.
- Personalized PageRank ran a node-by-node Python power iteration and rebuilt
  the graph for every user.
- The ranking scripts scored one user per process and refit the model each time.
- The evaluator stored every user's full candidate list and its bootstrap
  helper scanned all rows once per user.

## Decision

1. All four baselines share one `InteractionMatrix` (`src/trustrec/recommenders/matrix.py`).
   It keeps the first event per user-item pair before the cutoff, as the row
   contract does. Users and items are indexed in ascending ID order, so index
   order is also the `item_id_ascending` tie-break order. Arrow tables and pandas
   frames use a vectorized path, and other iterables use the row contract. Both
   paths produce the same matrix fingerprint.
2. `MatrixRanker` (`src/trustrec/recommenders/base.py`) owns fitting, candidate
   validation, fallbacks, and top-K ordering. `rank` keeps the `RankingResult`
   contract. `recommend` returns the top K from the catalog minus history.
3. Item kNN computes one sparse co-occurrence product per user. Its cosine and
   summation order match the set-based formula bit for bit on the test fixtures.
4. PPR iterates item mass only, using `p_i(t) = d^2 M p_i(t-2) + d (1-d) v_r`.
   Users with one positive edge contribute a diagonal term. This equals the
   previous node-by-node iteration up to floating-point rounding, including
   the stopping rule over all user and item nodes.
5. BPR MF trains with mini-batch SGD and summed updates (`batch_size`, default
   1024), with rejection sampling of unknown items. This is a different
   optimizer from the previous per-pair SGD, so results are not bit-identical
   to earlier BPR runs. `batch_size` and the optimizer name are part of the
   configuration and the model hash.
6. Model hashes use the matrix fingerprint and, for BPR, a hash of the learned
   arrays, instead of serializing the full training data.
7. The evaluator (protocol `evaluation-v2`) holds one shared catalog and each
   user's history. A ranking row declares its candidate pool either with
   `candidate_item_ids` or with `candidate_rule`, `candidate_count`, and
   `candidate_signature`. The signature hashes the catalog hash with the user's
   sorted history. The candidate set hash covers the catalog hash and each
   user's history instead of full candidate lists.

## Consequences

`scripts/run_rankings_batch.py` fits one baseline once and writes top-K rows
for every evaluation user. Its rows feed `scripts/run_evaluation.py`.
`candidate_retention` becomes `K / candidate_count` for top-K rows.
`evaluation-v1` candidate set hashes are not comparable with `evaluation-v2`
hashes. H0 and T0 still need full-catalog component scores and aspect evidence
for every evaluation user, so they are not part of the batch script yet.
