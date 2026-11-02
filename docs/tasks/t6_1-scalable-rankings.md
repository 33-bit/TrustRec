# Task: T6.1: Scalable baselines and batch ranking files

Status: done for code, tests, and a synthetic full-size run. Full Video Games
results still require the external Parquet tables.

## Context and output

- Decision: [ADR-0014](../adr/0014-sparse-rankers-and-compact-candidates.md)
- Specs: [`ranking-contract.md`](../specs/ranking-contract.md) and [`evaluation-contract.md`](../specs/evaluation-contract.md)
- Roadmap: Phase 3 (ranking core) and Phase 5 (evaluation)

The task makes B0 through B3 and the T5.1 evaluator run on the full snapshot.
`scripts/run_rankings_batch.py` writes one JSON Lines ranking file and one run
manifest per model:

```bash
PYTHONPATH=src python3 scripts/run_rankings_batch.py \
  --snapshot-manifest data/manifests/video_games-full-d6c4efeb74aa.json \
  --model b2_bpr_mf --cutoff t1 --seed 7 --k 100 --workers 4 \
  --output artifacts/rankings/b2_t1_seed07.jsonl
```

Each row holds `user_id`, `ranked_item_ids`, `scores`, `candidate_rule`,
`candidate_count`, `candidate_signature`, `fallback_reason`, `latency_ms`, and
`fit_time_ms`. The manifest records the snapshot, dataset hash, data
fingerprint, catalog hash, cutoff, seed, parameters, configuration hash, model
hash, source hashes, fallback counts, timings, and output hash. A completed
manifest is never replaced.

## Synthetic full-size run

Measured on a 4-core container with a synthetic snapshot of 1,545,000 fit
interactions (about 630,000 active users and 49,700 items) and 76,000 target
rows, which give 54,994 evaluation users. Real data will differ.

| Step | Wall time |
| --- | --- |
| Build interaction matrix | about 12 s |
| B0 batch, 4 workers | 24 s |
| B1 batch, 4 workers | 333 s |
| B2 batch, 50 epochs, 4 workers | measurement pending |
| B3 scoring | about 0.3 to 0.4 s per user per core at `max_iter=100` |
| `run_evaluation.py` with B0 and B1 | 113 s, 2.8 GB peak memory |

B3 is bound by single-threaded sparse products. With 4 workers, about 55,000
users take roughly 1.5 hours. `--max-iter` changes the model and its hash, so
treat a smaller value as a new configuration.

## Acceptance criteria

- [x] A deterministic command reproduces the output.
- [x] Tests cover the changed contract: matrix paths, parity with the
  set-based kNN and node-by-node PPR references, negative sampling, signature
  validation, worker determinism, and the batch-to-evaluator path.
- [x] The output records its snapshot ID, cutoff, seed, and hashes.
- [x] Documentation and backlog status match the result.

## Not covered

- H0 and T0 batch rows. They need full-catalog component scores and aspect
  evidence for every user (T6.2 and T6.3).
- Full Video Games ranking and metric files (T6.4).
