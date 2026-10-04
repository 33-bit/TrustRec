# TrustRec delivery roadmap

Read [the terminology guide](../reference/terminology.md) for terms used in this plan. The [task backlog](../tasks/backlog.md) records status.

## Phase 0: Foundation

Create repository tools, contracts, ADRs, protocol settings, structural tests, and the contributor guide. Run `make validate` and the contract tests.

## Phase 1: Data and snapshots

Profile the Video Games category. Join reviews with metadata. Keep software games and exclude unrelated hardware. Create time-safe train, validation, and recommendation test tables. Record source and window hashes.

## Phase 2: Aspect NLP

Run the 100-unit LLM development pilot before `T0`. Freeze an independent `llm_pseudo_test` split. Report pseudo-label consistency and explanation audit results. Do not create a human gold set.

## Phase 3: Ranking core

Implement popularity, item kNN, BPR MF, PPR, percentile normalization, and fallbacks. Use one candidate and target set for each comparison.

## Phase 4: TrustRec and explanations

Aggregate weighted evidence and implement the adaptive gate. Add source links, abstention, and the explanation audit contract.

## Phase 5: Evaluation and demo

Run ablations, sparse-history slices, noise tests, and bootstrap intervals. Build the Streamlit flow. Link each claim to an immutable manifest.
