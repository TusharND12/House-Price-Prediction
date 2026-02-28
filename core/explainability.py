"""
SmartExplain AI - Feature Contribution Explainability
=====================================================
Contribution_i = w_i * x_i (per-feature contribution to prediction)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class FeatureExplainer:
    """
    Explains model predictions via linear feature contributions.
    For y = w·x + b, contribution_i = w_i * x_i
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            weights: Model weight vector (w)
            bias: Model bias (b)
            feature_names: Optional names for features
        """
        self.weights = weights
        self.bias = bias
        self.feature_names = feature_names or [f"x_{i}" for i in range(len(weights))]

    def explain(
        self, X: np.ndarray, index: Optional[int] = None
    ) -> dict:
        """
        Compute contribution breakdown for one or all samples.

        Args:
            X: Feature matrix (n_samples, n_features) or single row (n_features,)
            index: If X has multiple rows, explain this index. Otherwise explain all.

        Returns:
            dict with: total_prediction, contributions, percentages, feature_names
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if index is not None:
            X = X[index : index + 1]
        # Contribution_i = w_i * x_i (element-wise)
        contributions = X * self.weights
        total_per_sample = np.sum(contributions, axis=1) + self.bias
        # Percentage influence: |contribution_i| / sum(|contribution_j|) * 100
        abs_contrib = np.abs(contributions)
        denom = np.sum(abs_contrib, axis=1, keepdims=True)
        denom[denom == 0] = 1e-8
        percentages = 100 * abs_contrib / denom
        return {
            "total_prediction": total_per_sample,
            "contributions": contributions,
            "percentages": percentages,
            "feature_names": self.feature_names,
        }
