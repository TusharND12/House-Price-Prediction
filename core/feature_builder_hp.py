"""
Feature builder for High-Performance pipeline (notebook HP section).
Builds a single row matching: num_cols + log + extra + one-hot ocean_proximity.
Used by the API when serving the best-tuned model from the notebook.
"""

from __future__ import annotations

import numpy as np
from typing import Any


NUM_COLS = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
]
LOG_NAMES = ["log_rooms", "log_bedrooms", "log_population", "log_households", "log_income"]
EXTRA_NAMES = [
    "income_x_rooms", "income_x_lat", "lon_x_lat", "rooms_per_hh",
    "log_rooms_sq", "income_sq", "log_pop_x_income",
]


def build_features_from_api_input(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
    ohe: Any,
) -> np.ndarray:
    """
    Build one row of features matching the notebook HP pipeline.
    ohe: fitted OneHotEncoder (drop='first') from the notebook.
    Returns shape (1, n_features).
    """
    lon = float(longitude)
    lat = float(latitude)
    age = float(housing_median_age)
    tr = float(total_rooms)
    bed = float(total_bedrooms)
    pop = float(population)
    hh = float(households) + 1.0
    mi = float(median_income)
    mi = max(mi, 0.01)

    # Numerical (8)
    X_num = np.array([[lon, lat, age, tr, bed, pop, hh - 1.0, mi]])

    # Log (5)
    X_log = np.array([[
        np.log1p(tr),
        np.log1p(bed),
        np.log1p(pop),
        np.log1p(hh - 1.0),
        np.log1p(mi),
    ]])

    # Extra (7)
    X_extra = np.array([[
        mi * tr,
        mi * lat,
        lon * lat,
        tr / hh,
        np.log1p(tr) ** 2,
        mi ** 2,
        np.log1p(pop) * mi,
    ]])

    # One-hot ocean_proximity (drop='first')
    ohe_row = ohe.transform([[str(ocean_proximity).strip()]])

    return np.hstack([X_num, X_log, X_extra, ohe_row])
