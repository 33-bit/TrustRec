# ADR-0003: Python Research Stack

- **Status:** Accepted
- **Date:** 2026-10-02

## Decision

Use Python 3.11+, Parquet, and DuckDB for tabular data; sparse matrices for graph computation; scikit-learn for baseline NLP and kNN; PyTorch only when a neural extension is justified; and Streamlit for the first demo. Keep the serving path artifact-based.

## Rationale

This stack matches the course methods, supports columnar profiling without loading a full category into memory, and keeps operational work proportional to a 3–4 person capstone team. A graph database or separate frontend is not required for the core evidence.

