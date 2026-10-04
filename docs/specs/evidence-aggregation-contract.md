# Evidence Aggregation Contract

Evidence is source text that supports a claim. Aggregation combines evidence from several reviews. For item `i`, aspect `a`, and cutoff `τ`, each review contributes at most once after clause aggregation.

The default review weight is:

```text
q_ra(τ) = confidence_ra × duplicate_penalty_r × recency_r(τ)
```

Shrinkage moves a score toward a training reference. The item aspect score uses these formulas:

```text
v_ia = (λ μ_a + Σ q_ra z_ra) / (λ + Σ q_ra)
C_ia = Σ q_ra / (λ + Σ q_ra)
```

`C_ia` describes support, not a confidence interval. Store positive, negative, and neutral mass separately. Also store the distinct author count and effective sample size. Effective sample size measures the concentration of evidence weights.

If evidence is missing, keep the prior score and set support to 0. If evidence conflicts, show the disagreement. Do not select only favorable sentences.

Select claims by user aspect weight × item deviation from the prior. Apply support thresholds. Avoid repeated passages from one author. A refusal reason is a valid explanation output.
