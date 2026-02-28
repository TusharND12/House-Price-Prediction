"""
SmartExplain AI - Linear Regression with Gradient Descent
=========================================================
Implements linear regression from scratch using various gradient descent variants.
Mathematical formulation: y = Xw + b

Cost Function (L2 regularized):
    J(w,b) = (1/2m) Σ(y_pred - y)² + λΣw²

Gradient Updates:
    dw = (1/m) Xᵀ(y_pred - y) + 2λw
    db = (1/m) Σ(y_pred - y)
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional
from .optimizers import MomentumOptimizer, LearningRateScheduler


class LinearRegressionGD:
    """
    Linear Regression trained via Gradient Descent.
    Supports Batch GD, Mini-batch GD, SGD with optional momentum and LR decay.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: Optional[int] = None,
        mode: Literal["batch", "minibatch", "sgd"] = "batch",
        regularization: float = 0.01,
        use_momentum: bool = False,
        momentum: float = 0.9,
        use_lr_decay: bool = False,
        decay_type: Literal["time", "step", "exponential"] = "time",
        early_stopping: bool = True,
        patience: int = 50,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the Linear Regression GD model.

        Args:
            learning_rate: Initial learning rate (α)
            n_iterations: Maximum number of gradient descent iterations
            batch_size: Batch size for minibatch/SGD. None = full batch
            mode: "batch" (full), "minibatch", or "sgd" (batch_size=1)
            regularization: L2 regularization strength (λ)
            use_momentum: Whether to use momentum optimizer
            momentum: Momentum coefficient (β)
            use_lr_decay: Whether to decay learning rate over time
            decay_type: "time", "step", or "exponential"
            early_stopping: Stop if cost doesn't improve
            patience: Early stopping patience (iterations)
            tol: Convergence tolerance for cost change
            random_state: Seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.mode = mode
        self.regularization = regularization
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.use_lr_decay = use_lr_decay
        self.decay_type = decay_type
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.cost_history: list[float] = []
        self._n_features: Optional[int] = None
        self._feature_names: Optional[list[str]] = None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Compute y_pred = Xw + b (linear combination)."""
        return X @ self.weights + self.bias

    def _compute_cost(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """
        Cost function: J(w,b) = (1/2m) Σ(y_pred - y)² + λΣw²
        MSE term + L2 penalty
        """
        m = len(y)
        mse_term = 0.5 * np.mean((y_pred - y) ** 2)
        l2_penalty = self.regularization * np.sum(self.weights**2)
        return float(mse_term + l2_penalty)

    def _compute_gradients(
        self, X_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients: dw = (1/m) Xᵀ(y_pred - y) + 2λw, db = (1/m) Σ(y_pred - y)
        """
        m = X_batch.shape[0]
        error = y_pred - y_batch
        # dw = (1/m) Xᵀ * error + 2λw (derivative of λw² w.r.t w)
        dw = (X_batch.T @ error) / m + 2 * self.regularization * self.weights
        db = np.mean(error)
        return dw, db

    def _get_batch_indices(self, m: int, iteration: int) -> np.ndarray:
        """Get indices for current batch (for minibatch/SGD)."""
        rng = np.random.default_rng(
            self.random_state + iteration if self.random_state is not None else None
        )
        indices = rng.permutation(m)
        batch_size = self.batch_size or m
        batch_idx = indices[:batch_size]
        return batch_idx

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> LinearRegressionGD:
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional names for interpretability

        Returns:
            self for method chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        m, n = X.shape
        self._n_features = n
        self._feature_names = feature_names or [f"x_{i}" for i in range(n)]

        # Initialize weights: w ~ small random values, b = 0
        self.weights = np.zeros(n)
        self.bias = 0.0

        # Set batch size based on mode
        if self.mode == "sgd":
            self.batch_size = 1
        elif self.mode == "batch":
            self.batch_size = m
        elif self.batch_size is None:
            self.batch_size = min(32, m)

        # Optimizer components
        velocity = np.zeros_like(self.weights) if self.use_momentum else None
        momentum_opt = MomentumOptimizer(self.momentum) if self.use_momentum else None
        lr_scheduler = (
            LearningRateScheduler(
                initial_lr=self.learning_rate,
                decay_type=self.decay_type,
                n_iterations=self.n_iterations,
            )
            if self.use_lr_decay
            else None
        )

        best_cost = float("inf")
        no_improvement_count = 0
        self.cost_history = []

        for iteration in range(self.n_iterations):
            # Get learning rate (possibly decayed)
            lr = (
                lr_scheduler.get_lr(iteration)
                if lr_scheduler
                else self.learning_rate
            )

            # Get batch
            batch_idx = self._get_batch_indices(m, iteration)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # Forward pass
            y_pred = self._predict(X_batch)

            # Compute gradients
            dw, db = self._compute_gradients(X_batch, y_batch, y_pred)

            # Apply momentum (if enabled)
            if momentum_opt is not None and velocity is not None:
                velocity = momentum_opt.update(velocity, dw)
                dw = velocity

            # Gradient clipping (prevent overflow)
            grad_norm = np.linalg.norm(dw)
            if grad_norm > 1e6:
                dw = dw * (1e6 / grad_norm)
            if abs(db) > 1e6:
                db = np.sign(db) * 1e6

            # Gradient descent update: w := w - α*dw, b := b - α*db
            self.weights -= lr * dw
            self.bias -= lr * db

            # Record cost (on full batch for monitoring)
            y_full_pred = self._predict(X)
            cost = self._compute_cost(y_full_pred, y)
            self.cost_history.append(cost)

            # Early stopping
            if self.early_stopping:
                if cost < best_cost:
                    best_cost = cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.patience:
                        break
                if iteration > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values: y_pred = Xw + b."""
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._predict(X)

    def get_params(self) -> dict:
        """Return model parameters for persistence."""
        return {
            "weights": self.weights.copy() if self.weights is not None else None,
            "bias": self.bias,
            "n_features": self._n_features,
            "feature_names": self._feature_names,
        }
