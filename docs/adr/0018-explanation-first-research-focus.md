# ADR-0018: Explanation-first research focus

Status: Accepted by the project owner on 2026-11-06. Refines [ADR-0001](0001-scope-and-trust-semantics.md). Revises the research questions in [`research-scope.md`](../specs/research-scope.md).

## Context

The full Video Games evaluation (T6.4) answered the accuracy questions:

- B5 recent popularity (60 days) reaches NDCG@10 0.1064 on `t1`, about four
  times B0. Every personalized model loses to it unless it adds a recency term.
- With a recency term (ADR-0017), H0 reaches 0.1071, +0.0007 over B5. Most of
  that lead comes from the graph term.
- The aspect term adds about +0.0001 NDCG@10. That gain disappears when its
  weight moves to graph instead. The T0 gate has no measurable effect. LLM and
  dictionary evidence give the same gain.

Aspect evidence does not measurably improve ranking accuracy on this data. It
can still let each recommendation carry claims that a user can trace to source
reviews, which is what ADR-0001 defines as TrustRec.

## Decision

1. Make evidence-grounded explanation the primary contribution. Accuracy
   becomes a guardrail, not the main claim.
2. The recommender that explanations attach to is the ADR-0017 H0 selected on
   `t0` (MF 0, graph 0.2, recency 0.6, aspect 0.2, 60-day window). Do not
   reselect it.
3. Accuracy guardrail: report that recommender against B5 on `t1` with a paired
   bootstrap interval. Call it "on par with B5" only if the whole 95% interval
   of the NDCG@10 difference lies within ±0.005. Call it "better than B5" only
   if the interval is above 0. This margin was set after the T6.4 results were
   known. It is used only to word the claim, never to choose a model.
4. Keep the T6.4 accuracy results, including null and negative ones, in the
   record. Report them as findings (recency dominates; aspect adds no robust
   accuracy) and do not hide them.
5. The new primary questions are about explanations: coverage, abstention,
   faithfulness, and claim correctness (see the research scope).
6. Explanations use only the curated LLM evidence (`curated/gpt-6-luna`).
   The T2.3 dictionary labels stay in the code and in past results as an NLP
   baseline and as the T6.4 robustness check. They are not used for new
   explanation work. Comparing evidence sources for explanations is an
   extension, not a primary question. The owner added this point on
   2026-11-07, before any explanation run.

## Consequences

- Explanation metrics must be measured on the real snapshot, for the top-K of
  the chosen recommender. Synthetic fixture results do not count.
- The explanation pipeline must not change the ranking. An explanation run
  that reranks items is a different recommender and needs its own accuracy row.
- Reports must not claim that aspect evidence improves accuracy.
