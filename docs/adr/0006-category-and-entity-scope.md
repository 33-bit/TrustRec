# ADR-0006: Category and Entity Scope

- **Status:** Accepted for pilot
- **Date:** 2026-10-03

## Decision

Use Video Games as the primary category. Filter the review category to software game entities after joining metadata; classify accessories, consoles, controllers, cables, headsets, keyboards, sellers, shipping, and packaging as out of scope for game aspects. Keep Electronics as a fallback only if the deterministic join and retention check invalidate Video Games.

## Consequences

The annotation pipeline must inspect item metadata before accepting a review as a game-aspect example. A category file name alone is not enough. All discarded scope decisions are counted and reported.

