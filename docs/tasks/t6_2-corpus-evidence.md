# Task: T6.2: Aspect evidence for the full review corpus

Status: done for the pipeline, the dictionary labeler, and synthetic
full-size runs. The real corpus run needs the external snapshot tables.
[ADR-0015](../adr/0015-rule-based-corpus-evidence.md) leaves open whether
rule-based evidence may feed reported T0 results.

## Context and output

- Decisions: [ADR-0013](../adr/0013-llm-only-annotation-policy.md) and [ADR-0015](../adr/0015-rule-based-corpus-evidence.md)
- Specs: [`evidence-aggregation-contract.md`](../specs/evidence-aggregation-contract.md) and [`nlp-annotation-contract.md`](../specs/nlp-annotation-contract.md)
- Roadmap: Phase 4 (TrustRec and explanations)

`scripts/label_corpus_evidence.py` writes one evidence JSON Lines file and one
manifest per cutoff. `scripts/build_evidence_profiles.py` turns that file into
item-aspect profiles.

```bash
PYTHONPATH=src python3 scripts/label_corpus_evidence.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --cutoff t1 --workers 4 \
  --output artifacts/evidence/t1_dictionary.jsonl
PYTHONPATH=src python3 scripts/build_evidence_profiles.py \
  --evidence artifacts/evidence/t1_dictionary.jsonl \
  --snapshot-id video_games-full-d6c4efeb74aa \
  --dataset-hash 0b990d349f917c8b864a17a729f73f83314829b78d7307073d6bd0a644404043 \
  --cutoff-timestamp 2020-08-31T22:46:09.247000+00:00 \
  --output artifacts/evidence/t1_profiles.json
```

Each evidence row holds `review_id`, `item_id`, `author_id`, `timestamp`,
`duplicate_group`, `aspect`, `sentiment`, `sentiment_score`, `confidence`,
`start`, `end`, `evidence`, `source_text_sha256`, and `label_source`.
`build_evidence_profiles.py` now streams rows and keeps only the fields that
profiles use. Its output is byte-identical to the previous version.

## Synthetic runs

Measured on 4 cores with synthetic reviews of about 120 characters. Real
reviews are longer, so expect longer labeling times.

| Reviews | Evidence rows | Labeling, 4 workers | Profiles | Peak profile memory |
| --- | --- | --- | --- | --- |
| 100,000 | 199,554 | 17 s | 16 s | 0.64 GB |
| 500,000 | 1,000,444 | 61 s | 41 s | 1.21 GB |

## Acceptance criteria

- [x] A deterministic command reproduces the output. Worker runs are byte-identical to single-process runs.
- [x] Tests cover the contract: exact offsets, text-hash checks, ratings ignored, refusals, cutoff and interaction filtering, the profile hand-off, and the manifest no-overwrite rule.
- [x] The output records its snapshot ID, cutoff, labeler hash, and source hashes.
- [x] Documentation and backlog status match the result.

## Real corpus runs (2026-11-02)

### Dictionary evidence

| Cutoff | Reviews | Labeled reviews | Evidence rows | Wall time, 2 workers |
| --- | --- | --- | --- | --- |
| `t0` | 1,373,677 | 543,204 | 1,155,985 | 624 s |
| `t1` | 1,545,387 | 586,355 | 1,227,183 | 663 s |

The outputs are in ignored `artifacts/evidence/{t0,t1}_dictionary.jsonl`.

### DeepSeek LLM labels

The project owner chose an LLM labeler. `scripts/label_reviews_llm.py` labels
every review before `T1` once; the `t0` evidence is the subset before `T0`.

- Model: `deepseek-flash` through the DeepSeek API. Batches of 20 reviews, temperature 0, prompt `t6.2-llm-batch-v1`.
- The reasoning setting was chosen on the `development_pilot` split (100 units). That split's labels came from another LLM, so the scores below are agreement, not accuracy.

  | Setting | Aspect micro F1 | Aspect-polarity F1 | Completion tokens |
  | --- | --- | --- | --- |
  | reasoning off (`reasoning_effort = none`) | 0.828 | 0.728 | 11,986 |
  | `reasoning_effort = low` | 0.842 | 0.717 | 72,313 |

  Reasoning off was chosen: agreement is about the same, with six times fewer tokens and about seven times faster requests.
- Process note: the same two settings were first scored on the frozen `llm_pseudo_test` split (reasoning off 0.842 / 0.802, low 0.880 / 0.851). ADR-0013 forbids using that split for model selection. Those numbers are reported only as consistency checks, and the choice above rests on the development pilot.
- Coverage: 18,537 of 1,545,387 reviews (1.2%) are labeled. 12,064 of them have at least one aspect label. All 922 requests succeeded. The run stopped when the account ran out of balance.
- Usage: 2,346,872 prompt tokens (236,288 cache hits) and 1,321,276 completion tokens, which is about 127 input and 72 output tokens per review. The measured cost was about 0.42 CNY per 1,000 reviews. The remaining reviews would cost about 640 CNY and take about 5 hours with 32 parallel requests.
- Cache: `artifacts/llm/deepseek_t1_labels.jsonl` and its `.runs.jsonl` log. The owner holds a downloaded copy. Rerunning the same command resumes from the cache.

Ways to reduce cost:

- Cap the number of reviews per item. At most 50 reviews per item leaves 526,309 reviews.
- Use a shorter JSON output format.
- Use larger batches.

Each of these changes the prompt or the sample, so it needs a new prompt version and a development-pilot check.
