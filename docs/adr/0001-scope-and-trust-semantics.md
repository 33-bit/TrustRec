# ADR-0001: Scope and Trust Semantics

- **Status:** Accepted
- **Date:** 2026-10-02

## Decision

TrustRec is an evidence-aware recommender. “Trust” means that a recommendation claim can be checked against source evidence and that the aggregate exposes support and disagreement. It does not mean that a reviewer is genuine or that a PageRank score proves honesty.

The core task is top-K ranking for known items using historical ratings, aspect opinions, and an interaction graph. Fraud detection, purchase-probability claims, and real-world business lift are outside the core evaluation.

## Consequences

Every explanation stores source IDs, text, aspect, sentiment, timestamp, and support metadata. Reports must use language bounded by the measured protocol.

