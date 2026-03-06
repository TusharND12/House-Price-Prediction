"""
Train the High-Performance pipeline (same as notebook HP section) and save model_hp.pkl
for the backend. Run from project root: python train_hp.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "housing.csv"


def main():
    if not DATA_PATH.exists():
        print("housing.csv not found. Place it in data/.")
        return
    df = pd.read_csv(DATA_PATH)
    df_hp = df.copy()
    df_hp["total_bedrooms"] = df_hp["total_bedrooms"].fillna(df_hp["total_bedrooms"].median())

    def remove_iqr(data, column, factor=1.5):
        q1, q3 = data[column].quantile(0.25), data[column].quantile(0.75)
        iqr = q3 - q1
        return data[(data[column] >= q1 - factor * iqr) & (data[column] <= q3 + factor * iqr)]

    df_hp = remove_iqr(df_hp, "median_house_value", 1.5)
    for col in ["median_income", "total_rooms", "population"]:
        df_hp = remove_iqr(df_hp, col, 2.0)

    ohe = OneHotEncoder(drop="first", sparse_output=False)
    ohe_cols = ohe.fit_transform(df_hp[["ocean_proximity"]])
    ohe_names = [f"ocean_{c}" for c in ohe.categories_[0][1:]]

    num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                "population", "households", "median_income"]
    X_num = df_hp[num_cols].values
    X_log = np.column_stack([
        np.log1p(df_hp["total_rooms"]),
        np.log1p(df_hp["total_bedrooms"]),
        np.log1p(df_hp["population"]),
        np.log1p(df_hp["households"]),
        np.log1p(df_hp["median_income"].clip(lower=0.01)),
    ])
    log_names = ["log_rooms", "log_bedrooms", "log_population", "log_households", "log_income"]
    mi = df_hp["median_income"].values
    tr = df_hp["total_rooms"].values
    lat = df_hp["latitude"].values
    lon = df_hp["longitude"].values
    hh = df_hp["households"].values + 1
    X_extra = np.column_stack([
        mi * tr, mi * lat, lon * lat, tr / hh, np.log1p(tr) ** 2,
        df_hp["median_income"].values ** 2,
        np.log1p(df_hp["population"].values) * mi,
    ])
    extra_names = ["income_x_rooms", "income_x_lat", "lon_x_lat", "rooms_per_hh", "log_rooms_sq", "income_sq", "log_pop_x_income"]

    X_eng = np.hstack([X_num, X_log, X_extra, ohe_cols])
    feature_names = num_cols + log_names + extra_names + ohe_names
    y = df_hp["median_house_value"].values

    X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
    except ImportError:
        pass

    best_model = None
    best_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        r2 = r2_score(y_test, model.predict(X_test_s))
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    out_path = Path(__file__).parent / "model_hp.pkl"
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "ohe": ohe,
        "feature_names": feature_names,
        "dataset": "california",
    }, out_path)
    print(f"Best model: {best_name} | Test R² = {best_r2:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
