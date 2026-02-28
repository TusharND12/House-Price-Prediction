"""
SmartExplain AI - What-If Price Simulator
=========================================
Simulate price change when changing a feature value.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def simulate_price_change(
    model,
    X_current: np.ndarray,
    feature_name: str,
    new_value: float,
    feature_names: list[str],
    explainer,
) -> dict:
    """
    Simulate prediction when one feature is changed.

    Args:
        model: Fitted LinearRegressionGD with predict()
        X_current: Current feature vector (1, n_features) - standardized
        feature_name: Name of feature to change
        new_value: New raw value (before standardization)
        feature_names: List of feature names
        explainer: FeatureExplainer instance

    Returns:
        dict with: original_prediction, updated_prediction, price_difference,
                   contribution_breakdown_original, contribution_breakdown_updated
    """
    if feature_name not in feature_names:
        raise ValueError(f"Unknown feature: {feature_name}")
    idx = feature_names.index(feature_name)
    # For raw new_value we need to standardize - but simulator receives
    # already-standardized X. We interpret new_value as the NEW standardized value
    # (user can pass standardized value from sliders).
    X_updated = X_current.copy()
    if X_updated.ndim == 1:
        X_updated = X_updated.reshape(1, -1)
    X_updated[0, idx] = new_value

    original_pred = model.predict(X_current.reshape(1, -1) if X_current.ndim == 1 else X_current)[0]
    updated_pred = model.predict(X_updated)[0]
    price_diff = updated_pred - original_pred

    expl_orig = explainer.explain(X_current)
    expl_upd = explainer.explain(X_updated)

    return {
        "original_prediction": float(original_pred),
        "updated_prediction": float(updated_pred),
        "price_difference": float(price_diff),
        "contribution_breakdown_original": {
            n: float(v) for n, v in zip(expl_orig["feature_names"], expl_orig["contributions"][0])
        },
        "contribution_breakdown_updated": {
            n: float(v) for n, v in zip(expl_upd["feature_names"], expl_upd["contributions"][0])
        },
        "percentage_influence_original": {
            n: float(v) for n, v in zip(expl_orig["feature_names"], expl_orig["percentages"][0])
        },
        "percentage_influence_updated": {
            n: float(v) for n, v in zip(expl_upd["feature_names"], expl_upd["percentages"][0])
        },
    }
