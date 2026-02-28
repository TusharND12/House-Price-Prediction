"""
SmartExplain AI - Feature Engineering for Kaggle House Prices Dataset
=====================================================================
Handles train.csv from house-prices-advanced-regression-techniques.
Uses numeric features only for reliability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


NUMERIC_COLS = [
    "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars",
    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]


class HousePricesFeatureEngineer:
    """Feature engineering for Kaggle House Prices - numeric features + polynomial."""

    def __init__(
        self,
        polynomial_degree: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        self.polynomial_degree = polynomial_degree
        self.random_state = random_state
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._feature_names: Optional[list[str]] = None
        self._n_poly: int = 0
        self._numeric_cols_used: Optional[list[str]] = None
        self._medians: Optional[np.ndarray] = None

    def _add_polynomial(self, X: np.ndarray, names: list[str], n_cols: int = 8) -> tuple[np.ndarray, list[str]]:
        k = min(n_cols, X.shape[1])
        new_cols = []
        new_names = list(names)
        for i in range(k):
            new_cols.append((X[:, i] ** 2).reshape(-1, 1))
            new_names.append(f"{names[i]}_sq")
        for i in range(k):
            for j in range(i + 1, k):
                new_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
                new_names.append(f"{names[i]}_x_{names[j]}")
        return np.hstack([X] + new_cols), new_names

    def fit_transform(self, df: pd.DataFrame, target_col: str = "SalePrice") -> tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        available = [c for c in NUMERIC_COLS if c in df.columns]
        self._numeric_cols_used = available
        num_df = df[available].copy()
        for c in num_df.columns:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce").fillna(num_df[c].median())
        X = num_df.values.astype(float)
        names = list(available)

        self._mean = np.mean(X, axis=0)
        self._medians = np.median(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1e-8
        self._n_poly = min(8, len(available)) if self.polynomial_degree >= 2 else 0
        X = (X - self._mean) / self._std

        if self._n_poly > 0:
            X, names = self._add_polynomial(X, names, n_cols=self._n_poly)
        self._feature_names = names

        y = df[target_col].values.astype(float)
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._mean is None:
            raise ValueError("Call fit_transform first.")
        df = df.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
        available = [c for c in NUMERIC_COLS if c in df.columns]
        num_df = df[available].copy()
        for c in num_df.columns:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce").fillna(num_df[c].median())
        X = num_df.values.astype(float)
        X = (X - self._mean) / self._std
        if self._n_poly > 0:
            X, _ = self._add_polynomial(X, list(available), n_cols=self._n_poly)
        return X

    def get_feature_names(self) -> list[str]:
        return self._feature_names or []

    def build_from_sliders(
        self,
        lot_area: float = 8450,
        overall_qual: float = 7,
        gr_liv_area: float = 1710,
        garage_cars: float = 2,
        total_bsmt_sf: float = 856,
        year_built: float = 2003,
        full_bath: float = 2,
        fireplace: float = 0,
    ) -> np.ndarray:
        """Build feature vector from key slider values for Streamlit app."""
        if self._mean is None or self._numeric_cols_used is None:
            raise ValueError("Call fit_transform first.")
        # Use training medians (or mean) as defaults, override with slider values
        m = getattr(self, "_medians", None)
        fallback = m if m is not None else self._mean
        defaults = {c: float(fallback[i]) for i, c in enumerate(self._numeric_cols_used)}
        defaults.update({
            "LotArea": lot_area, "OverallQual": overall_qual, "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars, "TotalBsmtSF": total_bsmt_sf, "YearBuilt": year_built,
            "FullBath": full_bath, "Fireplaces": fireplace,
        })
        row = np.array([[defaults.get(c, 0.0) for c in self._numeric_cols_used]])
        row = (row - self._mean[:row.shape[1]]) / self._std[:row.shape[1]]
        if self._n_poly > 0:
            row, _ = self._add_polynomial(row, self._numeric_cols_used, n_cols=self._n_poly)
        return row
