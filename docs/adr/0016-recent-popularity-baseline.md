# ADR-0016: Recent-popularity baseline (B5)

Status: Accepted. Date: 2026-11-05.

## Context

On the real `t1` test, H0 and T0 beat B0 by about 0.01 NDCG@10, while every
H0 and T0 setting was below popularity on `t0` validation. The whole gain came
from users without positive history. Those users got one shared list that kept
popular items with good aspect evidence and swapped out items popular only by
all-time count for items released close to `T1`. All-time popularity is a
stale signal, so the test gain may come from recency, not from aspect evidence.

## Decision

Add B5, `b5_recent_popularity`: rank candidates by positive first events in the
`window_days` days before the cutoff. Events at or after the cutoff never count.
Ties, including items with no event in the window, break by all-time popularity
and then by ascending item ID.

- B5 is not personalized and reads no review text or aspect evidence.
- `window_days` is tuned on `t0` only, with the 9-trial budget the other
  baselines received: 30, 60, 90, 180, 365, 540, 730, 1,095, and 1,825 days.
- The interaction matrix keeps each positive first event's timestamp. Its data
  fingerprint does not change, so saved models stay valid.

## Consequences

Reports compare H0 and T0 with B5 as well as B0. An H0 or T0 gain on users
without history counts as aspect evidence only if it also beats B5. B4 keeps
its meaning in the ranking contract (aspect-only score).
