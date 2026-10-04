# ADR-0003: Python Research Stack

Status: Accepted
Date: 2026-10-02

## Decision

Use Python 3.11 or newer, Parquet, and DuckDB for tables. Use sparse matrices for graph work. Use scikit-learn for NLP and kNN baselines. Use PyTorch only for a justified neural extension. Use Streamlit for the first demo. The serving path reads prepared artifacts.

## Rationale

This stack matches the course methods. It supports columnar profiling without loading a full category into memory. It also fits a 3–4 person capstone team. The core system does not need a graph database or a separate frontend.
