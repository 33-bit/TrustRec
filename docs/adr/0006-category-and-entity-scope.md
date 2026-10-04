# ADR-0006: Category and Entity Scope

Status: Accepted for pilot
Date: 2026-10-03

## Decision

Use Video Games as the primary category. Join reviews to metadata. Keep software game entities for game aspects. Accept the recorded software item types in the snapshot configuration. Mark accessories, consoles, controllers, cables, headsets, keyboards, books, sellers, shipping, and packaging as out of scope. Keep Electronics as a fallback for later comparison.

## Consequences

The annotation pipeline must inspect item metadata before it accepts a game-aspect example. A category file name is not enough. Count and report all scope exclusions.
