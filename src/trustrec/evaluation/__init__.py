"""Temporal evaluation, ranking metrics, and leakage checks."""

from .metrics import (
    BootstrapSummary,
    intra_list_diversity,
    ndcg_at_k,
    paired_user_bootstrap,
    precision_at_k,
    recall_at_k,
)
from .protocol import (
    EvaluationCase,
    EvaluationDataset,
    ModelEvaluation,
    build_evaluation_cases,
    compare_model_rankings,
)

__all__ = [
    "BootstrapSummary",
    "EvaluationCase",
    "EvaluationDataset",
    "ModelEvaluation",
    "build_evaluation_cases",
    "compare_model_rankings",
    "intra_list_diversity",
    "ndcg_at_k",
    "paired_user_bootstrap",
    "precision_at_k",
    "recall_at_k",
]
