# Ranking and Score Contract

## Modules

- B0: most popular from pre-cutoff positive interactions.
- B1: item kNN.
- B2: BPR MF with positive item and unknown negative sampling.
- B3: personalized PageRank on the positive user–item graph.
- B4: aspect-only score.
- H0: fixed normalized MF + graph + aspect hybrid.
- T0: adaptive TrustRec gate.

Normalize MF, graph, and aspect scores as percentile ranks within each user's candidate set. Ties receive the average rank; all-equal components receive a constant. These values are relative ranking scores, not probabilities.

```text
s_base = ρ M̃ + (1 - ρ) G̃
g(u,i) = g_max × κ/(κ+n_u) × C(u,i)
s_T0 = (1-g) s_base + g Ã
```

Candidate and target sets must be identical across compared models. Fallbacks for no history, no graph edge, and no evidence must be explicit in the output reason.

