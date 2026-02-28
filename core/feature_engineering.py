"""
SmartExplain AI - Feature Engineering
=====================================
Engineered features for house price prediction:
- area * location_rating
- age depreciation
- distance from city center
- interaction terms
- Log transforms (skewed features)
- Robust scaling (median/IQR) - outlier-resistant
- Outlier capping
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class FeatureEngineer:
    """
    Create and transform features for the California housing dataset.
    Handles standardization and engineered features.
    """

    def __init__(
        self,
        city_center_lon: float = -118.25,
        city_center_lat: float = 34.05,
        random_state: Optional[int] = None,
        use_robust_scaling: bool = True,
        use_log_transform: bool = True,
        cap_outliers: bool = True,
        outlier_percentile: tuple[float, float] = (1.0, 99.0),
        polynomial_degree: int = 2,
    ) -> None:
        """
        Args:
            city_center_lon: Longitude of reference city center (LA)
            city_center_lat: Latitude of reference city center
            random_state: For reproducibility
            use_robust_scaling: Use median/IQR instead of mean/std (outlier-resistant)
            use_log_transform: Log1p transform for skewed features
            cap_outliers: Cap feature values at percentiles
            outlier_percentile: (low, high) percentile for capping
            polynomial_degree: Degree for polynomial feature expansion (2 = squares + interactions)
        """
        self.city_center = (city_center_lon, city_center_lat)
        self.polynomial_degree = polynomial_degree
        self.random_state = random_state
        self.use_robust_scaling = use_robust_scaling
        self.use_log_transform = use_log_transform
        self.cap_outliers = cap_outliers
        self.outlier_percentile = outlier_percentile
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._median: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None
        self._feature_names: Optional[list[str]] = None
        self._ocean_proximity_map: Optional[dict] = None
        self._ocean_proximity_categories: Optional[list[str]] = None
        self._poly_indices: Optional[list] = None  # for build_from_sliders

    def _create_location_rating(self, df: pd.DataFrame) -> np.ndarray:
        """
        Map ocean_proximity to location_rating (1-5).
        NEAR BAY=5, <1H OCEAN=4, NEAR OCEAN=3, INLAND=2, ISLAND=5
        """
        if self._ocean_proximity_map is None:
            unique = df["ocean_proximity"].dropna().unique()
            rating_map = {
                "NEAR BAY": 5.0,
                "<1H OCEAN": 4.0,
                "NEAR OCEAN": 3.0,
                "INLAND": 2.0,
                "ISLAND": 5.0,
            }
            self._ocean_proximity_map = {
                k: rating_map.get(k, 3.0) for k in unique
            }
        return df["ocean_proximity"].map(self._ocean_proximity_map).fillna(3.0).values

    def _distance_from_center(self, df: pd.DataFrame) -> np.ndarray:
        """
        Euclidean distance from (longitude, latitude) to city center.
        d = sqrt((lon - lon_c)² + (lat - lat_c)²)
        """
        lon_c, lat_c = self.city_center
        lon = df["longitude"].values
        lat = df["latitude"].values
        return np.sqrt((lon - lon_c) ** 2 + (lat - lat_c) ** 2)

    def _age_depreciation(self, age: np.ndarray) -> np.ndarray:
        """
        Age depreciation factor: older houses = lower value multiplier.
        depreciation = exp(-0.02 * age) (mild decay)
        """
        return np.exp(-0.02 * age)

    def _add_polynomial_features(
        self, X: np.ndarray, names: list[str], degree: int, n_poly_cols: Optional[int] = None
    ) -> tuple[np.ndarray, list[str]]:
        """Add polynomial terms: x_i^2 and x_i*x_j for degree 2. n_poly_cols=first N cols only (key features)."""
        if degree < 2:
            return X, names
        n_features = X.shape[1]
        k = n_poly_cols if n_poly_cols is not None else n_features
        k = min(k, n_features)
        new_cols = []
        new_names = list(names)
        for i in range(k):
            new_cols.append((X[:, i] ** 2).reshape(-1, 1))
            new_names.append(f"{names[i]}_sq")
        for i in range(k):
            for j in range(i + 1, k):
                new_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
                new_names.append(f"{names[i]}_x_{names[j]}")
        X_poly = np.hstack([X] + new_cols)
        return X_poly, new_names

    def _get_one_hot(self, df: pd.DataFrame) -> np.ndarray:
        """One-hot encode ocean_proximity."""
        cats = df["ocean_proximity"].astype(str).unique().tolist()
        self._ocean_proximity_categories = sorted(cats)
        n = len(df)
        oh = np.zeros((n, len(self._ocean_proximity_categories)))
        for j, cat in enumerate(self._ocean_proximity_categories):
            oh[:, j] = (df["ocean_proximity"].astype(str) == cat).astype(float)
        return oh

    def fit_transform(self, df: pd.DataFrame, target_col: str = "median_house_value") -> tuple[np.ndarray, np.ndarray]:
        """
        Create engineered features and fit scaler, then transform.
        Preprocessing: outlier capping, log transforms, robust scaling.
        """
        df = df.copy()

        # Handle missing values
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

        # Base features (raw)
        area = df["total_rooms"].values.astype(float)
        bedrooms = df["total_bedrooms"].values.astype(float)
        population = df["population"].values.astype(float)
        households = df["households"].values.astype(float)
        median_income = df["median_income"].values.astype(float)
        age = df["housing_median_age"].values.astype(float)

        # Outlier capping - store bounds for transform()
        self._cap_bounds: Optional[dict[str, tuple[float, float]]] = None
        if self.cap_outliers:
            low_p, high_p = self.outlier_percentile
            raw_arrays = {"area": area, "bedrooms": bedrooms, "population": population,
                          "households": households, "income": median_income}
            self._cap_bounds = {}
            for name, arr in raw_arrays.items():
                lo, hi = np.percentile(arr, [low_p, high_p])
                self._cap_bounds[name] = (float(lo), float(hi))
                arr[:] = np.clip(arr, lo, hi)

        # Log transform for skewed features (improves linearity)
        if self.use_log_transform:
            log_area = np.log1p(area)
            log_bedrooms = np.log1p(bedrooms)
            log_population = np.log1p(population)
            log_households = np.log1p(households)
            log_income = np.log1p(median_income)
        else:
            log_area, log_bedrooms = area, bedrooms
            log_population, log_households = population, households
            log_income = median_income

        # Location rating and one-hot for ocean_proximity
        location_rating = self._create_location_rating(df)
        one_hot = self._get_one_hot(df)
        oh_names = [f"ocean_{c}" for c in (self._ocean_proximity_categories or [])]

        # Distance from city center + raw lat/lon (very predictive for CA housing)
        dist_center = self._distance_from_center(df)
        longitude = df["longitude"].values.astype(float)
        latitude = df["latitude"].values.astype(float)

        # Engineered features
        area_location = area * location_rating
        age_depr = self._age_depreciation(age)
        rooms_per_household = np.where(households > 0, area / households, 0)
        bedrooms_per_room = np.where(area > 0, bedrooms / area, 0)
        income_area = median_income * area
        income_age = median_income * age_depr
        dist_location = dist_center * location_rating

        # Build feature matrix (longitude, latitude are highly predictive for CA housing)
        base_features = [
            longitude, latitude, median_income, area, bedrooms, age, population, households,
            area_location, age_depr, dist_center,
            rooms_per_household, bedrooms_per_room, income_area,
            income_age, dist_location,
        ]
        base_names = [
            "longitude", "latitude", "median_income", "total_rooms", "total_bedrooms",
            "housing_median_age", "population", "households",
            "area_location_rating", "age_depreciation", "distance_from_center",
            "rooms_per_household", "bedrooms_per_room", "income_area",
            "income_age", "dist_location",
        ]
        if self.use_log_transform:
            base_features.extend([log_area, log_bedrooms, log_population, log_households, log_income])
            base_names.extend(["log_rooms", "log_bedrooms", "log_population", "log_households", "log_income"])

        features = np.column_stack(base_features)
        features = np.hstack([features, one_hot])
        all_names = base_names + oh_names

        # Robust scaling: (x - median) / IQR, fallback to mean/std
        self._mean = np.mean(features, axis=0)
        self._std = np.std(features, axis=0)
        self._std[self._std == 0] = 1e-8
        self._median = np.median(features, axis=0)
        q25, q75 = np.percentile(features, [25, 75], axis=0)
        self._iqr = q75 - q25
        self._iqr[self._iqr == 0] = 1e-8

        if self.use_robust_scaling:
            X = (features - self._median) / self._iqr
        else:
            X = (features - self._mean) / self._std

        # Polynomial expansion (degree 2: squares + pairwise products for key features only)
        if self.polynomial_degree >= 2:
            X, all_names = self._add_polynomial_features(X, all_names, self.polynomial_degree, n_poly_cols=10)
        self._feature_names = all_names
        y = df[target_col].values

        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler and feature engineering."""
        if self._mean is None or self._std is None:
            raise ValueError("Call fit_transform first.")

        df = df.copy()
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

        area = df["total_rooms"].values.astype(float)
        bedrooms = df["total_bedrooms"].values.astype(float)
        population = df["population"].values.astype(float)
        households = df["households"].values.astype(float)
        median_income = df["median_income"].values.astype(float)
        age = df["housing_median_age"].values.astype(float)

        if self.cap_outliers and self._cap_bounds:
            area = np.clip(area, *self._cap_bounds["area"])
            bedrooms = np.clip(bedrooms, *self._cap_bounds["bedrooms"])
            population = np.clip(population, *self._cap_bounds["population"])
            households = np.clip(households, *self._cap_bounds["households"])
            median_income = np.clip(median_income, *self._cap_bounds["income"])

        location_rating = self._create_location_rating(df)
        dist_center = self._distance_from_center(df)
        area_location = area * location_rating
        age_depr = self._age_depreciation(age)
        rooms_per_household = np.where(households > 0, area / households, 0)
        bedrooms_per_room = np.where(area > 0, bedrooms / area, 0)
        income_area = median_income * area
        income_age = median_income * age_depr
        dist_location = dist_center * location_rating

        longitude = df["longitude"].values.astype(float)
        latitude = df["latitude"].values.astype(float)
        base_features = [
            longitude, latitude, median_income, area, bedrooms, age, population, households,
            area_location, age_depr, dist_center,
            rooms_per_household, bedrooms_per_room, income_area,
            income_age, dist_location,
        ]
        if self.use_log_transform:
            base_features.extend([
                np.log1p(area), np.log1p(bedrooms), np.log1p(population),
                np.log1p(households), np.log1p(median_income),
            ])
        one_hot = np.zeros((len(df), len(self._ocean_proximity_categories or [])))
        for j, cat in enumerate(self._ocean_proximity_categories or []):
            one_hot[:, j] = (df["ocean_proximity"].astype(str) == cat).astype(float)
        features = np.hstack([np.column_stack(base_features), one_hot])
        if self.use_robust_scaling:
            X = (features - self._median) / self._iqr
        else:
            X = (features - self._mean) / self._std
        if self.polynomial_degree >= 2:
            base_names = [
                "longitude", "latitude", "median_income", "total_rooms", "total_bedrooms",
                "housing_median_age", "population", "households",
                "area_location_rating", "age_depreciation", "distance_from_center",
                "rooms_per_household", "bedrooms_per_room", "income_area",
                "income_age", "dist_location",
            ]
            if self.use_log_transform:
                base_names.extend(["log_rooms", "log_bedrooms", "log_population", "log_households", "log_income"])
            all_names = base_names + [f"ocean_{c}" for c in (self._ocean_proximity_categories or [])]
            X, _ = self._add_polynomial_features(X, all_names, self.polynomial_degree, n_poly_cols=10)
        return X

    def get_feature_names(self) -> list[str]:
        """Return list of feature names."""
        return self._feature_names or []

    def build_from_sliders(
        self,
        total_rooms: float,
        total_bedrooms: float,
        housing_median_age: float,
        median_income: float,
        population: float,
        households: float,
        location_rating: float,
        distance_from_center: float,
        ocean_proximity: str = "INLAND",
        longitude: float = -119.0,
        latitude: float = 36.0,
    ) -> np.ndarray:
        """
        Build standardized feature vector from user-provided values.
        Used by Streamlit app sliders.
        """
        if self._mean is None or self._std is None:
            raise ValueError("Call fit_transform first.")
        area = float(total_rooms)
        bedrooms = float(total_bedrooms)
        age = float(housing_median_age)
        pop = float(population)
        hh = float(households)
        inc = float(median_income)
        dist = float(distance_from_center)
        lon = float(longitude)
        lat = float(latitude)
        if self.cap_outliers and self._cap_bounds:
            area = np.clip(area, *self._cap_bounds["area"])
            bedrooms = np.clip(bedrooms, *self._cap_bounds["bedrooms"])
            pop = np.clip(pop, *self._cap_bounds["population"])
            hh = np.clip(hh, *self._cap_bounds["households"])
            inc = np.clip(inc, *self._cap_bounds["income"])
        area_location = area * location_rating
        age_depr = np.exp(-0.02 * age)
        rooms_per_household = area / hh if hh > 0 else 0
        bedrooms_per_room = bedrooms / area if area > 0 else 0
        income_area = inc * area
        income_age = inc * age_depr
        dist_location = dist * location_rating
        base_row = [
            lon, lat, inc, area, bedrooms, age, pop, hh,
            area_location, age_depr, dist,
            rooms_per_household, bedrooms_per_room, income_area,
            income_age, dist_location,
        ]
        if self.use_log_transform:
            base_row.extend([
                np.log1p(area), np.log1p(bedrooms), np.log1p(pop),
                np.log1p(hh), np.log1p(inc),
            ])
        cats = self._ocean_proximity_categories or []
        oh = [1.0 if str(c) == str(ocean_proximity) else 0.0 for c in cats]
        features = np.array([base_row + oh])
        if self.use_robust_scaling:
            X = (features - self._median) / self._iqr
        else:
            X = (features - self._mean) / self._std
        if self.polynomial_degree >= 2:
            base_names = [
                "longitude", "latitude", "median_income", "total_rooms", "total_bedrooms",
                "housing_median_age", "population", "households",
                "area_location_rating", "age_depreciation", "distance_from_center",
                "rooms_per_household", "bedrooms_per_room", "income_area",
                "income_age", "dist_location",
            ]
            if self.use_log_transform:
                base_names.extend(["log_rooms", "log_bedrooms", "log_population", "log_households", "log_income"])
            oh_names = [f"ocean_{c}" for c in cats]
            all_names = base_names + oh_names
            X, _ = self._add_polynomial_features(X, all_names, self.polynomial_degree, n_poly_cols=10)
        return X

    def get_scaler_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) for manual scaling."""
        if self._mean is None or self._std is None:
            raise ValueError("Call fit_transform first.")
        return self._mean.copy(), self._std.copy()
