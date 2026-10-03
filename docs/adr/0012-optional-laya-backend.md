# ADR-0012: Optional Local Laya Backend

- **Status:** Accepted for comparison only
- **Date:** 2026-10-03

## Decision

Laya remains an optional local silver-label backend. It is not the primary annotation source after the pilot comparison. The core test harness does not depend on Laya or its checkpoint weights.

## Consequences

Installing Laya is an explicit extra. Its outputs are retained for error comparison and triage, but the LLM development labels and human-reviewed policy drive the main NLP pipeline.

