# Ranking and Score Contract

Ranking puts candidate items in score order. The pipeline compares these models:

- B0: Most popular from positive interactions before the cutoff.
- B1: Item kNN.
- B2: BPR MF with positive items and sampled unknown negatives.
- B3: Personalized PageRank on the positive user-item graph.
- B4: Aspect-only score.
- H0: Fixed hybrid of normalized MF, graph, and aspect scores.
- T0: Adaptive TrustRec gate.

A percentile rank gives a score position within a set. Normalize MF, graph, and aspect scores within each user candidate set. Give ties their average rank. Give all-equal components a constant value. These are relative ranking scores, not probabilities.

```text
s_base = ρ M̃ + (1 - ρ) G̃
g(u,i) = g_max × κ/(κ+n_u) × C(u,i)
s_T0 = (1-g) s_base + g Ã
```

Use identical candidate and target sets across compared models. If history, graph edges, or evidence are absent, state the fallback reason in the output.
