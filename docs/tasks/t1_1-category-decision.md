# T1.1: category profile and decision

Date: 2026-10-04

Status: Complete for the full Video Games category profile.

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) by McAuley Lab. The reviewable profile is [`t1_1_category_profile.json`](t1_1_category_profile.json).

## Method

The pilot profile reads fixed byte windows. The full profile scans every review and metadata record with sequential range requests. It records source sizes, every chunk hash, full source hashes, missing fields, scope counts, temporal cutoffs, and retention.

The scope filter joins reviews to metadata before it keeps software game item types. It keeps `Video Game`, `Software Download`, `Game`, `Computer Game`, `CD-ROM`, `DVD-ROM`, `Game Cartridge`, and `Software`. It removes accessories, consoles, controllers, books, and other non-game products.

## Full Video Games result

The full scan reads 4,624,615 reviews and 137,269 metadata rows. The filter keeps 1,737,112 review rows before stable-ID deduplication. The final snapshot contains 1,717,097 interactions, 49,749 items, and 1,031,539 users. Metadata coverage is 100%.

The review text missing rate is 0.0257%. The mean non-empty review length is 393.06 characters. Ratings contain 1,071,120 five-star rows, 252,381 four-star rows, 132,926 three-star rows, 82,002 two-star rows, and 198,683 one-star rows.

The temporal cutoffs are `T0 = 2019-01-13T04:24:39.377Z` and `T1 = 2020-08-31T22:46:09.247Z`. The validation table contains 95,559 targets for 75,414 users. The recommendation test contains 76,435 targets for 61,775 users.

## Decision

Use Video Games as the primary full category. The full snapshot has enough users and targets for the planned ranking evaluation. Keep Electronics as a fallback for a separate comparison.

The full profile is complete. The recommendation benchmark still needs model runs and result reports.
