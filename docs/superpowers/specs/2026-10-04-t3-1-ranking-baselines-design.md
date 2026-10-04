# T3.1 Ranking Baselines Design

## Goal

Implement deterministic most-popular and item kNN baselines that rank the
same pre-cutoff candidate set and expose auditable score parts.

## Shared contract

`CandidateSet` carries the user ID, snapshot ID, cutoff timestamp, candidate
item IDs, and the user's pre-cutoff item history. Candidate IDs are unique
and cannot overlap with history IDs. The helper that builds a candidate set
uses items seen before the cutoff and removes every item in the user's
history, including low-rated items.

`RankingResult` carries the model ID, user and snapshot IDs, cutoff, the
candidate IDs used, ranked `RankedItem` records, configuration, seed, model
hash, and an optional fallback reason. Each ranked item carries its item ID,
rank, total score, and component scores. Rankers preserve the candidate set
and use item ID as the final tie break.

## Popularity baseline

`PopularityRanker` fits the first event for each user-item pair with a
timestamp strictly before the
cutoff. It counts ratings at or above the positive threshold, which defaults
to 4. It scores every known supplied candidate, including candidates with
count zero, then sorts by descending count and ascending item ID. It rejects
reviewed items and items that were unknown before the cutoff. An empty
candidate set returns an empty result with `no_candidates`. The model hash
covers the filtered positive counts and configuration.

## Item kNN baseline

`ItemKNNRanker` builds binary item-to-user sets from first pre-cutoff
user-item events whose rating is positive. For each candidate, it sums cosine
similarity to the user's positive history items. An optional neighbor limit
keeps the strongest history similarities for each candidate. The ranker uses
positive popularity as a tie break and returns both kNN and popularity
components. It rejects reviewed items and items that were unknown before the
cutoff.

If the user has no positive pre-cutoff history, item kNN falls back to the
same popularity counts and records `no_positive_history`. If no candidate
has a positive similarity, it uses the same fallback and records
`no_similar_items`. These fallbacks keep the output shape and candidate set
stable. They do not treat missing feedback as a negative label.

The reproducible command reads a prepared Parquet snapshot from its manifest.
It writes a JSON ranking artifact with the snapshot ID, dataset hash, cutoff,
configuration hash, model hash, and ranked result.

## Testing

Tests use fixed in-memory interactions and timestamps. They cover strict
cutoff filtering, candidate exclusion, stable ties, zero-count candidates,
cosine scoring, neighbor limits, empty candidates, and the item kNN
no-positive-history fallback. The full test suite and Ruff checks provide the
final verification.
