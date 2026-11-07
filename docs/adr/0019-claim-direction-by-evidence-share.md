# ADR-0019: Claim direction by the share of polar evidence

Status: Accepted by the project owner on 2026-11-07. Refines [ADR-0010](0010-explanation-abstention.md).

## Context

The first real explanation run (T7.1) marked 615,042 of 615,572 supported
claims as mixed. A claim was mixed whenever its (item, aspect) profile had any
positive and any negative mass. Recommended items are popular: the median
selected (item, aspect) pair has 91 evidence rows, so almost every pair has
some opposing reviews. About 80% of those "mixed" claims had at least two
thirds positive evidence. A "mixed" label on nearly every item tells the user
nothing.

## Decision

1. Let the positive share be the positive mass divided by the positive plus
   negative mass of the (item, aspect) profile. The masses already include
   confidence and duplicate weights. Neutral mass does not count.
2. The claim is positive if the share is at least `direction_share` (default
   2/3), negative if it is at most `1 - direction_share`, and mixed between
   those two bounds. A profile with no polar mass is neutral.
3. Positive and negative claims state the share, for example "83% of opinions
   are positive". ADR-0010 requires that conflicting evidence stays visible, so
   such a claim cites two passages that agree with it and, when one exists, one
   passage from the other side. A mixed claim cites both sides and leads with
   the side that has more passages.
4. `direction_share` is a convention, not tuned on test data. T7.3 audits
   whether the stated direction matches the cited evidence. A change to the
   default needs a new decision.

## Consequences

- Claims carry a `positive_share` field.
- `AggregatedEvidence.conflicting` still reports any two-sided evidence, but
  it no longer decides the claim direction.
- T7.1 results from before this ADR stay in the record as the first run.
