# ADR-0001: Scope and Trust Semantics

Status: Accepted
Date: 2026-10-02

## Decision

TrustRec is a recommender with evidence. A user can trace a claim to source review text. The system also shows support and disagreement. “Trust” does not mean that a reviewer is honest. A PageRank score does not prove honesty.

The core task ranks known items with ratings, aspect opinions, and a user-item graph. Fraud detection, purchase probabilities, and business lift are outside the core evaluation.

## Consequences

Each explanation stores source IDs, text, aspect, sentiment, timestamp, and support data. Reports must use claims that match the measured protocol.
