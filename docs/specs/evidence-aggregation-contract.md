# Evidence Aggregation Contract

For item `i`, aspect `a`, and cutoff `τ`, each review contributes at most once after clause aggregation. The default review weight is:

```text
q_ra(τ) = confidence_ra × duplicate_penalty_r × recency_r(τ)
```

The item aspect score uses shrinkage toward the training prior:

```text
v_ia = (λ μ_a + Σ q_ra z_ra) / (λ + Σ q_ra)
C_ia = Σ q_ra / (λ + Σ q_ra)
```

`C_ia` is support, not a confidence interval. Store positive, negative, and neutral mass separately; store distinct author count and effective sample size. If evidence is missing, retain the prior and support 0. If evidence conflicts, expose the disagreement rather than selecting only favorable sentences.

## Evidence selection

Select claims using user aspect weight × item deviation from prior, subject to support thresholds. Avoid repeated passages from one author. A refusal reason is a valid explanation output.

