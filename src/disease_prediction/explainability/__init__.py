"""Explainability: coefficients, tree importance, permutation importance and SHAP."""

from disease_prediction.explainability.explain import (
    CAVEAT,
    ExplanationError,
    global_explanations,
    linear_coefficients,
    local_explanation,
    permutation_importance_scores,
    shap_explanations,
    tree_feature_importance,
    unwrap_base_pipeline,
)

__all__ = [
    "CAVEAT",
    "ExplanationError",
    "global_explanations",
    "linear_coefficients",
    "local_explanation",
    "permutation_importance_scores",
    "shap_explanations",
    "tree_feature_importance",
    "unwrap_base_pipeline",
]
