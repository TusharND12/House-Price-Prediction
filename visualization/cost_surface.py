"""
SmartExplain AI - 3D Cost Surface Visualization
================================================
Visualize J(w1,w2) and gradient descent path.
For visualization we fix all but 2 weights and vary w1, w2.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


def plot_cost_surface_3d(
    X: np.ndarray,
    y: np.ndarray,
    weight_idx1: int = 0,
    weight_idx2: int = 1,
    w_opt: Optional[np.ndarray] = None,
    path: Optional[list[tuple[float, float]]] = None,
    n_grid: int = 30,
) -> plt.Figure:
    """
    Plot 3D cost surface J(w1,w2) with optional GD path.
    Fixes all weights except weight_idx1 and weight_idx2 at optimal (or 0).
    """
    m, n = X.shape
    if w_opt is None:
        w_opt = np.zeros(n)
    w_base = w_opt.copy()
    w1_range = np.linspace(w_opt[weight_idx1] - 2, w_opt[weight_idx1] + 2, n_grid)
    w2_range = np.linspace(w_opt[weight_idx2] - 2, w_opt[weight_idx2] + 2, n_grid)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = w_base.copy()
            w[weight_idx1] = W1[i, j]
            w[weight_idx2] = W2[i, j]
            y_pred = X @ w
            Z[i, j] = 0.5 * np.mean((y_pred - y) ** 2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.8)
    ax.set_xlabel(f"w_{weight_idx1}")
    ax.set_ylabel(f"w_{weight_idx2}")
    ax.set_zlabel("Cost J(w)")
    ax.set_title("3D Cost Surface")

    if path:
        p1 = [t[0] for t in path]
        p2 = [t[1] for t in path]
        costs = []
        for (a, b) in path:
            w = w_base.copy()
            w[weight_idx1] = a
            w[weight_idx2] = b
            costs.append(0.5 * np.mean((X @ w - y) ** 2))
        ax.plot(p1, p2, costs, "r-o", markersize=4, linewidth=1)
    return fig


def get_gradient_descent_path(
    X: np.ndarray,
    y: np.ndarray,
    weight_idx1: int,
    weight_idx2: int,
    n_steps: int = 20,
    lr: float = 0.1,
) -> list[tuple[float, float]]:
    """Simulate GD on 2 weights, return path for visualization."""
    m, n = X.shape
    w = np.zeros(n)
    path = []
    for _ in range(n_steps):
        path.append((float(w[weight_idx1]), float(w[weight_idx2])))
        y_pred = X @ w
        error = y_pred - y
        dw = X.T @ error / m
        w[weight_idx1] -= lr * dw[weight_idx1]
        w[weight_idx2] -= lr * dw[weight_idx2]
    return path
