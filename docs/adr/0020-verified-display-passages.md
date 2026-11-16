# ADR-0020: LLM-verified display passages

Status: Accepted by the project owner on 2026-11-07. Amends the citation rule in [ADR-0019](0019-claim-direction-by-evidence-share.md).

## Context

The T7.3 audit found that about 12% of cited passages, and 24% among the 30
most-shown claims, do not discuss the claimed aspect. Most are generic praise
or complaints that the labeler tagged as gameplay. Weighted by exposure,
about 60% of displayed claims were supported. The owner chose to verify each
passage with an LLM before display, not to filter passages with a length
heuristic.

## Decision

1. For each displayed claim, take candidate passages: up to 8 from the claim
   side and up to 4 from the opposing side, longest first. For a mixed claim,
   the claim side is the side with more passages.
2. A verifier model rates each candidate: does it discuss the aspect, what is
   its polarity, and is it about the reviewed product. The verifier does not
   see the labeler's polarity.
3. Display up to two passages that the verifier finds on-aspect, about the
   product, and on the claim side. Add one verified passage from the opposing
   side when one exists. A mixed claim needs one verified passage on each side.
4. When no passage on the claim side passes (for a mixed claim, when either
   side has none), the claim becomes `insufficient_support` with reason
   `no_verified_passage`. The system abstains instead of showing an
   unsupported claim.
5. Verification only changes which passages are shown and whether a claim is
   shown. It does not change scores, rankings, profiles, or claim direction.
6. Record the verifier model, prompt version, prompt hash, and temperature.
   Cache every verdict, so a stopped run resumes.

## Consequences

- When the same model family both verifies and audits, the audit agrees with
  itself. Reports must say so. An audit with a different judge model, or a
  human spot check, is the stronger test.
- Verification costs one LLM request per displayed claim. The cost scales
  with the number of distinct displayed claims, not with users.
