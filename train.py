"""
SmartExplain AI - Training Script
=================================
Load data, preprocess, train model, save with pickle.
Supports: Kaggle House Prices (train.csv) or California Housing (housing.csv).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd

from core.model import LinearRegressionGD
from core.metrics import mae, mse, rmse, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path(__file__).parent / "data"


def main():
    data_dir = DATA_DIR

    # Detect dataset: House Prices (train.csv) or California (housing.csv)
    train_path = data_dir / "train.csv"
    housing_path = data_dir / "housing.csv"

    if train_path.exists():
        print("Using Kaggle House Prices dataset (train.csv)...")
        df = pd.read_csv(train_path)
        from core.feature_engineering_house_prices import HousePricesFeatureEngineer

        fe = HousePricesFeatureEngineer(polynomial_degree=2, random_state=RANDOM_STATE)
        X, y = fe.fit_transform(df, target_col="SalePrice")
    elif housing_path.exists():
        print("Using California Housing dataset (housing.csv)...")
        df = pd.read_csv(housing_path)
        from core.feature_engineering import FeatureEngineer

        fe = FeatureEngineer(
            random_state=RANDOM_STATE,
            use_robust_scaling=False,
            use_log_transform=True,
            cap_outliers=True,
            polynomial_degree=2,
        )
        X, y = fe.fit_transform(df, target_col="median_house_value")
    else:
        raise FileNotFoundError("No dataset found. Add train.csv (House Prices) or housing.csv (California) to data/")

    feature_names = fe.get_feature_names()
    print(f"Features: {len(feature_names)}, Samples: {len(X)}")

    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    print("Training model (momentum + LR decay)...")
    model = LinearRegressionGD(
        learning_rate=0.03,
        n_iterations=5000,
        regularization=0.02,
        mode="batch",
        use_momentum=True,
        momentum=0.9,
        use_lr_decay=True,
        decay_type="time",
        early_stopping=True,
        patience=150,
        tol=1e-7,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, feature_names=feature_names)

    # Retrain on full data for production
    model_full = LinearRegressionGD(
        learning_rate=0.03,
        n_iterations=5000,
        regularization=0.02,
        mode="batch",
        use_momentum=True,
        momentum=0.9,
        use_lr_decay=True,
        decay_type="time",
        early_stopping=True,
        patience=200,
        tol=1e-7,
        random_state=RANDOM_STATE,
    )
    model_full.fit(X, y, feature_names=feature_names)
    model = model_full

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_full = model.predict(X)

    print("\nEvaluation metrics (Train 80%):")
    print("  MAE:  ", mae(y_train, y_pred_train))
    print("  RMSE: ", rmse(y_train, y_pred_train))
    print("  R2:   ", r2_score(y_train, y_pred_train))
    print("\nEvaluation metrics (Test):")
    print("  MAE:  ", mae(y_test, y_pred_test))
    print("  RMSE: ", rmse(y_test, y_pred_test))
    print("  R2:   ", r2_score(y_test, y_pred_test))
    print("\nEvaluation metrics (Full data - model for app):")
    print("  R2:   ", r2_score(y, y_pred_full))

    artifacts = {
        "model": model,
        "fe": fe,
        "feature_names": feature_names,
        "dataset": "house_prices" if train_path.exists() else "california",
    }
    out_path = Path(__file__).parent / "model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()
