# T1.1 — Category Profiling and Decision Memo

**Date:** 2026-10-02  
**Status:** Complete for pilot profiling; category decision remains subject to the T1.2 join/retention check.  
**Profile artifact:** [`t1_1_category_profile.json`](t1_1_category_profile.json)  
**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and the [Hugging Face dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## Method

The profile reads three deterministic 16 MiB byte windows (start, midpoint, end) from each category's raw review JSONL and metadata JSONL file. It parses complete records only and stores the source URL, byte size, window offsets, and sample SHA-256. This is a pilot profile: sampled unique counts and retention are estimates/lower bounds, not full-category statistics. The full-category counts below come from the dataset card.

## Results

| Category | Full reviews | Full users | Full items | Review file | Pilot review rows | Pilot text missing | Pilot validation retention | Pilot test retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video Games | 4.6M | 2.8M | 137.2K | 2.68 GB | 84,957 | 0.114% | 2,473 / 42,427 (5.83%) | 2,412 / 47,650 (5.06%) |
| Electronics | 43.9M | 18.3M | 1.6M | 22.62 GB | 89,681 | 0.122% | 1,804 / 33,300 (5.42%) | 1,748 / 36,454 (4.80%) |

The pilot uses global timestamp quantiles near 80% and 90%, rating ≥4 as the target, known items before the cutoff as the catalog, and the user's prior history to remove seen items. Video Games clears the planning target of roughly 2,000 eligible users in both windows on this pilot; Electronics does not.

Rating-positive share (4–5 stars) is 73.2% for Video Games and 75.9% for Electronics. Review text is present in 99.89% and 99.88% of pilot rows respectively. Median non-empty review length is 135 characters for Video Games and 125 for Electronics.

## Metadata and scope risks

Metadata missingness is similar for both categories: `description` 37.9%/40.9%, `features` 29.1%/25.2%, `price` 55.0%/66.9%, and `main_category` 22.1%/18.9% (Video Games/Electronics). `title` and `images` are nearly complete. `brand` is absent in all sampled metadata records; it is not a usable feature until the schema is verified. The category label in metadata is multi-valued/subcategory-like, so the review-file category must remain the primary scope filter.

## Decision

Use **Video Games** as the primary category for T1.2. It is approximately 9.5× smaller by review count and 8.4× smaller by raw review bytes than Electronics, while the pilot retains enough eligible users for the planned validation/test analysis and supports the proposed gameplay/story/performance ontology. Keep **Electronics** as the fallback only if the T1.2 metadata join shows that Video Games contains too many non-software items or loses the retention threshold after deterministic filtering.

## Next action

T1.2 created the first deterministic Video Games subset and matched all sampled item IDs to metadata; see [`t1_2-snapshot-report.md`](t1_2-snapshot-report.md). The next step is to classify software games versus accessories and add the leakage assertions in T1.3. Do not treat the pilot estimates as final benchmark counts.
