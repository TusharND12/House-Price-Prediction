"""
SmartExplain AI - Optimization Utilities
========================================
Momentum optimizer and learning rate decay schedulers.
"""

from __future__ import annotations

import numpy as np
from typing import Literal


class MomentumOptimizer:
    """
    Momentum optimizer: v = β*v + ∇J, then w = w - α*v
    Smooths gradient updates and accelerates convergence in consistent directions.
    """

    def __init__(self, momentum: float = 0.9) -> None:
        """
        Args:
            momentum: Decay factor (β) for previous velocity. Typically 0.9.
        """
        self.momentum = momentum

    def update(self, velocity: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Update velocity: v = β*v + gradient
        Returns the updated velocity (to be used as -α*v for weight update).
        """
        return self.momentum * velocity + gradient


class LearningRateScheduler:
    """
    Learning rate decay scheduler.
    Reduces learning rate over iterations for finer convergence.
    """

    def __init__(
        self,
        initial_lr: float,
        decay_type: Literal["time", "step", "exponential"],
        n_iterations: int,
        decay_rate: float = 0.01,
        step_size: int = 100,
    ) -> None:
        """
        Args:
            initial_lr: Starting learning rate (α₀)
            decay_type: "time" (1/(1+decay*t)), "step" (step decay), "exponential"
            n_iterations: Total iterations (for scaling)
            decay_rate: Decay parameter
            step_size: For step decay, reduce every step_size iterations
        """
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.n_iterations = n_iterations
        self.decay_rate = decay_rate
        self.step_size = step_size

    def get_lr(self, iteration: int) -> float:
        """
        Get learning rate at given iteration.

        Time decay: α_t = α₀ / (1 + decay_rate * t)
        Step decay: α_t = α₀ * decay_rate^(t // step_size)
        Exponential: α_t = α₀ * exp(-decay_rate * t)
        """
        t = iteration
        if self.decay_type == "time":
            # α_t = α₀ / (1 + decay_rate * t)
            return self.initial_lr / (1.0 + self.decay_rate * t)
        elif self.decay_type == "step":
            # α_t = α₀ * decay_rate^floor(t / step_size)
            return self.initial_lr * (self.decay_rate ** (t // self.step_size))
        elif self.decay_type == "exponential":
            # α_t = α₀ * exp(-decay_rate * t)
            return self.initial_lr * np.exp(-self.decay_rate * t)
        else:
            return self.initial_lr
