# ADR-0005: Aspect Evidence and Evaluation Contract

Status: Accepted
Date: 2026-10-02

## Decision

The first game ontology uses gameplay, story, graphics, performance, controls, multiplayer, content-and-replay, and value. Store evidence at sentence or clause level. Store positive, negative, or neutral sentiment and confidence. Start evaluation with most popular, item kNN, BPR MF, PPR, aspect-only, fixed hybrid, and adaptive TrustRec.

## Consequences

Report NLP quality separately from recommendation quality. Measure explanation coverage and correctness. Include cases where the system refuses a strong explanation. Optional models cannot replace the baseline comparison.
