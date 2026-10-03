# TrustRec Task Matrix

| Workstream | Primary files | Artifact | Verification | Related RQ |
| --- | --- | --- | --- | --- |
| Data source/profile | `scripts/profile_categories.py` | category memo/profile JSON | source bytes, missingness, retention | feasibility |
| Snapshot | `scripts/build_snapshot.py` | manifest + Parquet tables | repeat hashes | all |
| Leakage | `src/trustrec/evaluation/leakage.py` | leakage report | timestamp/ID assertions | all |
| Annotation | `configs/aspects.toml`, `docs/tasks/*annotation*` | pilot/gold JSONL | offset and split checks | RQ3/RQ4 |
| NLP baseline | `src/trustrec/nlp/` | evidence table + metrics | F1/coverage | RQ1/RQ3 |
| Collaborative ranking | `src/trustrec/recommenders/` | model artifact + scores | same candidates | RQ1 |
| Graph ranking | `src/trustrec/graph/` | PPR scores + graph manifest | popularity comparison | RQ1 |
| Evidence/profile | `src/trustrec/data/`, `src/trustrec/explanations/` | profiles + evidence | support/duplicate checks | RQ2/RQ3 |
| Hybrid gate | `src/trustrec/recommenders/` | fixed/adaptive runs | validation tuning | RQ2 |
| Faithfulness | `src/trustrec/explanations/` | removal audit | contribution delta | RQ4 |
| Evaluation | `src/trustrec/evaluation/` | metrics/bootstrap/slices | manifest lineage | all |
| Demo/report | `app/`, `reports/` | demo + figures/report | five-minute script | RQ4 |

