# LLM label dataset

This spec defines the curated LLM aspect labels that
`scripts/curate_llm_labels.py` writes and that the private Hugging Face
dataset `33bit/trustrec` stores. The labels are `llm_silver` evidence
([ADR-0013](../adr/0013-llm-only-annotation-policy.md)). They are not human gold
labels.

## Layout

| Path | Content |
| --- | --- |
| `curated/<model>/reviews.parquet` | One row per labeled review |
| `curated/<model>/labels.parquet` | One row per aspect label |
| `curated/<model>/manifest.json` | Provenance, counts, cleaning counts, and file hashes |
| `llm/<model>/` | Raw append-only cache, kept for resume and audit |

Use the `curated/` tables for analysis and for the evidence pipeline. The raw
cache holds failed attempts, retries, and per-row run metadata.

## `reviews.parquet`

| Column | Type | Meaning |
| --- | --- | --- |
| `review_id` | string | Snapshot review ID; the primary key |
| `source_text_sha256` | string | SHA-256 of the labeled review text |
| `status` | string | `labeled`, `no_aspect`, `out_of_scope`, or `empty_text` |
| `needs_review` | bool | The model marked the review unclear or too short |
| `truncated` | bool | The text sent to the model was cut at 4,000 characters |
| `n_labels` | int32 | Rows for this review in `labels.parquet` |
| `dropped_labels` | int32 | Model labels rejected at parse time (unknown aspect or no exact span) |

Rows are sorted by `review_id`.

## `labels.parquet`

| Column | Type | Meaning |
| --- | --- | --- |
| `review_id` | string | Foreign key to `reviews.parquet` |
| `label_index` | int32 | Position of the label in its review, from 0 |
| `aspect` | string | `gameplay`, `story`, `graphics`, `performance`, `controls`, `multiplayer`, `content_replay`, or `value` |
| `polarity` | string | `positive`, `negative`, or `neutral` |
| `start`, `end` | int32 | Character offsets into the original review text |
| `evidence` | string | Exactly `text[start:end]` |

Rows are sorted by `review_id` and then by `start`, `end`, `aspect`, and
`polarity`.

## Privacy

The tables hold no reviewer, author, or item identities. Join on `review_id`
with the snapshot to get `item_id`, `author_id`, and `timestamp`. The
`evidence` column holds review excerpts, so the dataset must stay private.

## Curation rules (`llm-curation-v1`)

1. Keep the last successful record per review for one model, reasoning
   setting, and prompt version. Drop failed attempts.
2. Keep only reviews in the scope's interaction table whose text hash still
   matches the snapshot.
3. Require `evidence == text[start:end]` for every label. A mismatch stops
   curation.
4. Trim leading and trailing whitespace from each span.
5. If one aspect has both `positive` and `negative` on the same span, write one
   `neutral` label, because the claim is mixed.
6. Merge overlapping spans that have the same aspect and polarity into their
   union, so one claim is not counted twice.
7. Drop labels on `out_of_scope` reviews.

`manifest.json` records how often each rule fired (`raw_cache.label_actions`),
the SHA-256 of every raw cache part, and the SHA-256 of both tables.

## Rebuild and use

```bash
PYTHONPATH=src python3 scripts/curate_llm_labels.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --cache artifacts/llm/gpt-6-luna --model gpt-6-luna \
  --output artifacts/curated/gpt-6-luna \
  --hf-repo 33bit/trustrec --hf-prefix curated/gpt-6-luna
PYTHONPATH=src python3 scripts/label_corpus_evidence.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --cutoff t1 --workers 4 --labeler llm-cache \
  --label-cache artifacts/curated/gpt-6-luna \
  --output artifacts/evidence/t1_gpt-6-luna.jsonl
```

Use `--reasoning-effort none` for the DeepSeek cache. With a curated folder,
`label_corpus_evidence.py` reads the model from `manifest.json` and checks
both table hashes before it uses the labels.
