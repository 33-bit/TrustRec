# Evaluation and Ablation Contract

## Temporal protocol

Use global `T0 < T1`. Train uses `<T0`; validation targets use `[T0,T1)`; fit-final uses `<T1`; test targets use `≥T1`. Exclude prior user items from candidates and require target items to be known before the relevant cutoff.

## Metrics

Primary: NDCG@10. Supporting: Recall@5/10/20, Precision@K, coverage, novelty, diversity, explanation correctness, explanation coverage, p50/p95 latency, RAM, and fit time. Report eligible users and excluded-user reasons.

## Required slices

User history 1–2, 3–5, and >5; item support groups; popularity groups; users with no history as a separately named fallback stress case. Use paired bootstrap over users for metric differences and three final seeds where the model is stochastic.

## Ablations

A1 remove graph; A2 remove aspect; A3 remove evidence weighting; A4 replace adaptive gate with fixed weights. Keep candidate sets, data, tuning budget, and seeds comparable.

## Leakage gates

Fail if feature timestamp ≥ cutoff, target review ID appears in feature/evidence, metadata is post-cutoff, models see different candidates, or a test result is used to tune a threshold.

## Manual-label evaluation gates

Amazon Reviews 2023 by McAuley Lab supplies training/recommendation interactions. Manual labels are held out from recommendation, aspect-extraction, and sentiment-model training. A development/pilot manual subset may refine guidelines, prompts, or settings; a separate final manual test subset is frozen before those choices and is used only for final aspect, sentiment, and explanation-quality evaluation. Manual labeling is sampled and does not cover the full corpus.
