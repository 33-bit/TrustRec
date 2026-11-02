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
