"""
SmartExplain AI - Plotting Utilities
====================================
Cost vs Iterations, Optimizer comparison, Actual vs Predicted, Learning rate comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_cost_vs_iterations(
    cost_history: list[float],
    title: str = "Cost vs Iterations",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot training cost over iterations."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history, color="steelblue", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J(w,b)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig or plt.gcf()


def plot_optimizer_comparison(
    histories: dict[str, list[float]],
    title: str = "Optimizer Comparison",
) -> plt.Figure:
    """Compare cost curves for different optimizers/configurations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for (name, history), c in zip(histories.items(), colors):
        ax.plot(history, label=name, color=c, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scatter plot of actual vs predicted values."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    return fig or plt.gcf()


def plot_learning_rate_comparison(
    lr_histories: dict[str, list[float]],
    title: str = "Learning Rate Comparison",
) -> plt.Figure:
    """Compare convergence for different learning rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lr_histories)))
    for (name, history), c in zip(lr_histories.items(), colors):
        ax.plot(history, label=name, color=c, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
