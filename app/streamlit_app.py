"""
SmartExplain AI - Streamlit Interactive App
===========================================
Sliders for Area, Bedrooms, Bathrooms, Location rating, Age, Distance.
Shows predicted price, contribution chart, learning curve, model comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

RANDOM_STATE = 42


def load_model_and_artifacts():
    """Load trained model, feature engineer, and metadata."""
    base = Path(__file__).parent.parent
    model_path = base / "model.pkl"
    if not model_path.exists():
        return None, None, None, None, None
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts.get("model"), artifacts.get("fe"), artifacts.get("feature_names"), artifacts.get("dataset")


def main():
    st.set_page_config(page_title="SmartExplain AI", page_icon="🏠", layout="wide")
    st.title("SmartExplain AI – House Price Prediction")

    model, fe, feature_names, dataset = load_model_and_artifacts()
    if model is None or fe is None:
        st.warning("Model not found. Run `python train.py` first.")
        return
    dataset = dataset or "california"

    # Sliders depend on dataset
    if dataset == "house_prices":
        st.caption("Kaggle House Prices dataset")
        col1, col2 = st.columns(2)
        with col1:
            lot_area = st.slider("Lot Area (sq ft)", 1000, 50000, 8450)
            overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
            gr_liv_area = st.slider("Above ground living area", 300, 6000, 1710)
            garage_cars = st.slider("Garage cars", 0, 4, 2)
        with col2:
            total_bsmt_sf = st.slider("Total basement sq ft", 0, 6000, 856)
            year_built = st.slider("Year built", 1880, 2010, 2003)
            full_bath = st.slider("Full bathrooms", 0, 4, 2)
            fireplace = st.slider("Fireplaces", 0, 4, 0)
        X = fe.build_from_sliders(
            lot_area=float(lot_area),
            overall_qual=float(overall_qual),
            gr_liv_area=float(gr_liv_area),
            garage_cars=float(garage_cars),
            total_bsmt_sf=float(total_bsmt_sf),
            year_built=float(year_built),
            full_bath=float(full_bath),
            fireplace=float(fireplace),
        )
    else:
        defaults = {"total_rooms": 2635, "total_bedrooms": 537, "housing_median_age": 29,
                    "median_income": 3.87, "population": 1425, "households": 499,
                    "location_rating": 3.5, "distance_from_center": 0.5,
                    "longitude": -119.0, "latitude": 36.0}
        st.caption("California Housing dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.slider("Area (total_rooms)", 100, 20000, int(defaults["total_rooms"]))
            bedrooms = st.slider("Bedrooms", 1, 2000, int(defaults["total_bedrooms"]))
        with col2:
            location_rating = st.slider("Location rating (1-5)", 1.0, 5.0, defaults["location_rating"], 0.1)
            age = st.slider("Housing median age", 1, 52, int(defaults["housing_median_age"]))
            dist = st.slider("Distance from city center", 0.0, 2.0, defaults["distance_from_center"], 0.05)
        with col3:
            median_income = st.slider("Median income", 0.5, 15.0, float(defaults["median_income"]), 0.1)
            population = st.slider("Population", 0, 35000, int(defaults["population"]))
            households = st.slider("Households", 0, 6000, int(defaults["households"]))
        longitude = st.slider("Longitude", -124.0, -114.0, defaults["longitude"], 0.1)
        latitude = st.slider("Latitude", 32.0, 42.0, defaults["latitude"], 0.1)
        ocean_proximity = st.selectbox("Ocean proximity", ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"], index=1)
        X = fe.build_from_sliders(
            total_rooms=float(area), total_bedrooms=float(bedrooms), housing_median_age=float(age),
            median_income=median_income, population=float(population), households=float(households),
            location_rating=location_rating, distance_from_center=dist,
            ocean_proximity=ocean_proximity, longitude=longitude, latitude=latitude,
        )

    pred = model.predict(X)[0]
    st.metric("Predicted price ($)", f"{pred:,.0f}")

    # Contribution bar chart
    from core.explainability import FeatureExplainer

    explainer = FeatureExplainer(model.weights, model.bias, feature_names)
    result = explainer.explain(X)
    contribs = result["contributions"][0]
    names = result["feature_names"]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    order = np.argsort(np.abs(contribs))[::-1][:12]
    ax1.barh([names[i] for i in order], contribs[order])
    ax1.set_xlabel("Contribution to prediction")
    ax1.set_title("Feature contributions")
    ax1.invert_yaxis()
    st.pyplot(fig1)
    plt.close()

    # Learning curve (if available)
    if hasattr(model, "cost_history") and model.cost_history:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(model.cost_history, color="steelblue")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cost")
        ax2.set_title("Training cost curve")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()

    # Model comparison placeholder
    st.subheader("Model comparison")
    st.caption("Our GD model vs Sklearn LinearRegression (run notebook for full comparison)")


if __name__ == "__main__":
    main()
