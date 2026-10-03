# ADR-0005: Aspect Evidence and Evaluation Contract

- **Status:** Accepted
- **Date:** 2026-10-02

## Decision

The initial game ontology uses gameplay, story, graphics, performance, controls, multiplayer, content-and-replay, and value. Evidence is stored at sentence or clause level with positive, negative, or neutral sentiment and calibrated confidence. The evaluation ladder begins with most popular, item kNN, BPR MF, PPR, aspect-only, fixed hybrid, and the adaptive TrustRec model.

## Consequences

NLP quality is reported separately from recommendation quality. Explanation coverage and correctness are both measured, including cases where the system refuses a strong explanation for insufficient support. Optional models cannot replace the agreed baseline comparison.

