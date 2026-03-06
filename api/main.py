"""
SmartExplain AI - FastAPI Backend
=================================
REST API for predictions, explainability, and simulation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SmartExplain AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent.parent


def load_artifacts():
    # Prefer best-tuned model from notebook (model_hp.pkl)
    hp_path = BASE / "model_hp.pkl"
    if hp_path.exists():
        try:
            import joblib
            return joblib.load(hp_path)
        except Exception:
            pass
    p = BASE / "model.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


ARTIFACTS = load_artifacts()


def _is_hp_artifacts():
    """True if loaded artifacts are from the notebook HP pipeline (best-tuned model)."""
    return ARTIFACTS is not None and "scaler" in ARTIFACTS and "ohe" in ARTIFACTS


def _build_hp_features(data: dict):
    """Build one row of features for HP model from API input."""
    from core.feature_builder_hp import build_features_from_api_input
    ohe = ARTIFACTS["ohe"]
    return build_features_from_api_input(
        longitude=float(data.get("longitude", -119)),
        latitude=float(data.get("latitude", 36)),
        housing_median_age=float(data.get("housing_median_age", 29)),
        total_rooms=float(data.get("total_rooms", 2635)),
        total_bedrooms=float(data.get("total_bedrooms", 537)),
        population=float(data.get("population", 1425)),
        households=float(data.get("households", 499)),
        median_income=float(data.get("median_income", 3.87)),
        ocean_proximity=str(data.get("ocean_proximity", "INLAND")),
        ohe=ohe,
    )


def _hp_predict_and_contributions(X_s: np.ndarray):
    """Return (prediction, contributions_breakdown) for HP model. X_s is scaled."""
    model = ARTIFACTS["model"]
    fn = ARTIFACTS["feature_names"]
    pred = float(model.predict(X_s)[0])
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        total = imp.sum() or 1e-10
        pcts = (imp / total * 100).tolist()
        contribs = (pred * imp / total).tolist()
    else:
        contribs = [0.0] * len(fn)
        pcts = [100.0 / len(fn)] * len(fn)
    breakdown = [{"name": fn[i], "contribution": contribs[i], "percent": pcts[i]} for i in range(len(fn))]
    breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return pred, breakdown


class HousePricesInput(BaseModel):
    lot_area: float = 8450
    overall_qual: float = 7
    gr_liv_area: float = 1710
    garage_cars: float = 2
    total_bsmt_sf: float = 856
    year_built: float = 2003
    full_bath: float = 2
    fireplace: float = 0


class CaliforniaInput(BaseModel):
    total_rooms: float = 2635
    total_bedrooms: float = 537
    housing_median_age: float = 29
    median_income: float = 3.87
    population: float = 1425
    households: float = 499
    location_rating: float = 3.5
    distance_from_center: float = 0.5
    longitude: float = -119.0
    latitude: float = 36.0
    ocean_proximity: str = "INLAND"


@app.get("/api/info")
def get_info():
    if not ARTIFACTS:
        return {"error": "Model not found. Run notebook HP section and save model_hp.pkl, or run train.py."}
    ds = ARTIFACTS.get("dataset", "house_prices")
    fn = ARTIFACTS.get("feature_names", [])
    if _is_hp_artifacts():
        return {"dataset": ds, "feature_names": fn, "n_features": len(fn), "model_r2": "~0.85", "model_type": "best_tuned"}
    return {
        "dataset": ds,
        "feature_names": fn,
        "n_features": len(fn),
        "model_r2": "~0.90" if ds == "house_prices" else "~0.70",
    }


@app.post("/api/predict")
def predict(data: dict):
    if not ARTIFACTS:
        return {"error": "Model not found."}
    try:
        if _is_hp_artifacts():
            X = _build_hp_features(data)
            X_s = ARTIFACTS["scaler"].transform(X)
            pred, breakdown = _hp_predict_and_contributions(X_s)
            return {"prediction": pred, "contributions": breakdown[:15], "cost_history": []}
        model = ARTIFACTS["model"]
        fe = ARTIFACTS["fe"]
        fn = ARTIFACTS["feature_names"]
        y_scaler = ARTIFACTS.get("y_scaler")
        ds = ARTIFACTS.get("dataset", "house_prices")
        if ds == "house_prices":
            X = fe.build_from_sliders(
                lot_area=float(data.get("lot_area", 8450)),
                overall_qual=float(data.get("overall_qual", 7)),
                gr_liv_area=float(data.get("gr_liv_area", 1710)),
                garage_cars=float(data.get("garage_cars", 2)),
                total_bsmt_sf=float(data.get("total_bsmt_sf", 856)),
                year_built=float(data.get("year_built", 2003)),
                full_bath=float(data.get("full_bath", 2)),
                fireplace=float(data.get("fireplace", 0)),
            )
        else:
            X = fe.build_from_sliders(
                total_rooms=float(data.get("total_rooms", 2635)),
                total_bedrooms=float(data.get("total_bedrooms", 537)),
                housing_median_age=float(data.get("housing_median_age", 29)),
                median_income=float(data.get("median_income", 3.87)),
                population=float(data.get("population", 1425)),
                households=float(data.get("households", 499)),
                location_rating=float(data.get("location_rating", 3.5)),
                distance_from_center=float(data.get("distance_from_center", 0.5)),
                ocean_proximity=str(data.get("ocean_proximity", "INLAND")),
                longitude=float(data.get("longitude", -119)),
                latitude=float(data.get("latitude", 36)),
            )
        raw_pred = float(model.predict(X)[0])
        if y_scaler is not None:
            pred = float(y_scaler.inverse_transform([[raw_pred]])[0, 0])
        else:
            pred = raw_pred
        from core.explainability import FeatureExplainer
        exp = FeatureExplainer(model.weights, model.bias, fn)
        res = exp.explain(X)
        contribs = res["contributions"][0]
        scale = float(getattr(getattr(y_scaler, "scale_", [1.0]), "__getitem__", lambda i: 1.0)(0)) if y_scaler is not None else 1.0
        contribs = (contribs * scale).tolist()
        pcts = res["percentages"][0].tolist()
        breakdown = [{"name": fn[i], "contribution": contribs[i], "percent": pcts[i]} for i in range(len(fn))]
        breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return {"prediction": pred, "contributions": breakdown[:15], "cost_history": getattr(model, "cost_history", [])}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/simulate")
def simulate(data: dict):
    """Compare two scenarios. Pass current and updated inputs."""
    if not ARTIFACTS:
        return {"error": "Model not found."}
    curr = data.get("current", {})
    upd = data.get("updated", curr)
    try:
        if _is_hp_artifacts():
            X1 = _build_hp_features(curr)
            X2 = _build_hp_features(upd)
            scaler = ARTIFACTS["scaler"]
            p1, br1 = _hp_predict_and_contributions(scaler.transform(X1))
            p2, br2 = _hp_predict_and_contributions(scaler.transform(X2))
            fn = ARTIFACTS["feature_names"]
            return {
                "original_prediction": p1,
                "updated_prediction": p2,
                "price_difference": p2 - p1,
                "contributions_original": [{"name": b["name"], "value": b["contribution"]} for b in br1[:10]],
                "contributions_updated": [{"name": b["name"], "value": b["contribution"]} for b in br2[:10]],
            }
        from core.explainability import FeatureExplainer
        model = ARTIFACTS["model"]
        fe = ARTIFACTS["fe"]
        fn = ARTIFACTS["feature_names"]
        y_scaler = ARTIFACTS.get("y_scaler")
        exp = FeatureExplainer(model.weights, model.bias, fn)
        ds = ARTIFACTS.get("dataset", "house_prices")
        hp_defaults = {"lot_area": 8450, "overall_qual": 7, "gr_liv_area": 1710, "garage_cars": 2, "total_bsmt_sf": 856, "year_built": 2003, "full_bath": 2, "fireplace": 0}
        cal_defaults = {"total_rooms": 2635, "total_bedrooms": 537, "housing_median_age": 29, "median_income": 3.87, "population": 1425, "households": 499, "location_rating": 3.5, "distance_from_center": 0.5, "ocean_proximity": "INLAND", "longitude": -119, "latitude": 36}
        if ds == "house_prices":
            c1 = {k: float(curr.get(k, hp_defaults.get(k, 0))) for k in ["lot_area", "overall_qual", "gr_liv_area", "garage_cars", "total_bsmt_sf", "year_built", "full_bath", "fireplace"]}
            c2 = {k: float(upd.get(k, hp_defaults.get(k, 0))) for k in ["lot_area", "overall_qual", "gr_liv_area", "garage_cars", "total_bsmt_sf", "year_built", "full_bath", "fireplace"]}
            X1 = fe.build_from_sliders(**c1)
            X2 = fe.build_from_sliders(**c2)
        else:
            c1 = {k: curr.get(k) if k == "ocean_proximity" else float(curr.get(k, cal_defaults.get(k, 0))) for k in ["total_rooms", "total_bedrooms", "housing_median_age", "median_income", "population", "households", "location_rating", "distance_from_center", "ocean_proximity", "longitude", "latitude"]}
            c2 = {k: upd.get(k) if k == "ocean_proximity" else float(upd.get(k, cal_defaults.get(k, 0))) for k in ["total_rooms", "total_bedrooms", "housing_median_age", "median_income", "population", "households", "location_rating", "distance_from_center", "ocean_proximity", "longitude", "latitude"]}
            X1 = fe.build_from_sliders(**c1)
            X2 = fe.build_from_sliders(**c2)
        r1_raw = float(model.predict(X1)[0])
        r2_raw = float(model.predict(X2)[0])
        if y_scaler is not None:
            p1 = float(y_scaler.inverse_transform([[r1_raw]])[0, 0])
            p2 = float(y_scaler.inverse_transform([[r2_raw]])[0, 0])
        else:
            p1, p2 = r1_raw, r2_raw
        r1 = exp.explain(X1)
        r2 = exp.explain(X2)
        return {
            "original_prediction": p1,
            "updated_prediction": p2,
            "price_difference": p2 - p1,
            "contributions_original": [{"name": fn[i], "value": float(r1["contributions"][0][i])} for i in range(min(len(fn), 10))],
            "contributions_updated": [{"name": fn[i], "value": float(r2["contributions"][0][i])} for i in range(min(len(fn), 10))],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/map-data")
def get_map_data():
    """Sample California housing data for map display."""
    p = BASE / "data" / "housing.csv"
    if not p.exists():
        return {"data": []}
    df = pd.read_csv(p).head(500)
    return {"data": df[["longitude", "latitude", "median_house_value"]].fillna(0).to_dict(orient="records")}


@app.get("/api/cost-history")
def cost_history():
    if not ARTIFACTS:
        return {"data": []}
    m = ARTIFACTS.get("model")
    h = getattr(m, "cost_history", [])
    return {"data": h}


# --- Never-Seen-Before Features ---

def _get_prediction_and_contributions(data: dict):
    """Helper: run prediction and return (pred, contributions, fn)."""
    if not ARTIFACTS:
        return None, [], []
    try:
        if _is_hp_artifacts():
            X = _build_hp_features(data)
            X_s = ARTIFACTS["scaler"].transform(X)
            pred, breakdown = _hp_predict_and_contributions(X_s)
            return pred, breakdown, ARTIFACTS["feature_names"]
        from core.explainability import FeatureExplainer
        model = ARTIFACTS["model"]
        fe = ARTIFACTS["fe"]
        fn = ARTIFACTS["feature_names"]
        y_scaler = ARTIFACTS.get("y_scaler")
        ds = ARTIFACTS.get("dataset", "house_prices")
        if ds == "house_prices":
            X = fe.build_from_sliders(
                lot_area=float(data.get("lot_area", 8450)),
                overall_qual=float(data.get("overall_qual", 7)),
                gr_liv_area=float(data.get("gr_liv_area", 1710)),
                garage_cars=float(data.get("garage_cars", 2)),
                total_bsmt_sf=float(data.get("total_bsmt_sf", 856)),
                year_built=float(data.get("year_built", 2003)),
                full_bath=float(data.get("full_bath", 2)),
                fireplace=float(data.get("fireplace", 0)),
            )
        else:
            X = fe.build_from_sliders(
                total_rooms=float(data.get("total_rooms", 2635)),
                total_bedrooms=float(data.get("total_bedrooms", 537)),
                housing_median_age=float(data.get("housing_median_age", 29)),
                median_income=float(data.get("median_income", 3.87)),
                population=float(data.get("population", 1425)),
                households=float(data.get("households", 499)),
                location_rating=float(data.get("location_rating", 3.5)),
                distance_from_center=float(data.get("distance_from_center", 0.5)),
                ocean_proximity=str(data.get("ocean_proximity", "INLAND")),
                longitude=float(data.get("longitude", -119)),
                latitude=float(data.get("latitude", 36)),
            )
        raw_pred = float(model.predict(X)[0])
        if y_scaler is not None:
            pred = float(y_scaler.inverse_transform([[raw_pred]])[0, 0])
        else:
            pred = raw_pred
        exp = FeatureExplainer(model.weights, model.bias, fn)
        res = exp.explain(X)
        contribs = res["contributions"][0]
        scale = float(getattr(getattr(y_scaler, "scale_", [1.0]), "__getitem__", lambda i: 1.0)(0)) if y_scaler is not None else 1.0
        contribs = contribs * scale
        breakdown = [{"name": fn[i], "contribution": float(contribs[i])} for i in range(len(fn))]
        breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return pred, breakdown, fn
    except Exception:
        return None, [], []


def _feature_story_category(name: str, ds: str) -> str:
    """Map feature name to story category."""
    n = name.lower()
    if ds == "house_prices":
        if any(x in n for x in ["lot", "grliv", "bsmt", "area", "garage", "year", "bath", "fireplace"]):
            return "structure"
        if any(x in n for x in ["qual", "cond"]):
            return "quality"
        return "lifestyle"
    # California
    if any(x in n for x in ["longitude", "latitude", "distance", "ocean", "location"]):
        return "location"
    if any(x in n for x in ["room", "bedroom", "area", "household"]):
        return "structure"
    return "lifestyle"


@app.post("/api/voice-explain")
def voice_explain(data: dict):
    """Generate natural-language text for text-to-speech."""
    pred, breakdown, _ = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "text": ""}
    top = breakdown[:5]
    parts = [f"The predicted price is ${pred:,.0f}."]
    for c in top:
        amt = abs(c["contribution"])
        if amt < 100:
            continue
        direction = "adds" if c["contribution"] >= 0 else "reduces"
        parts.append(f"{c['name'].replace('_', ' ')} {direction} about ${amt:,.0f} to the price.")
    return {"text": " ".join(parts), "prediction": pred}


@app.post("/api/story-decomposition")
def story_decomposition(data: dict):
    """Group contributions into Structure, Location/Lifestyle stories."""
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "stories": []}
    ds = ARTIFACTS.get("dataset", "house_prices")
    stories = {}
    for c in breakdown:
        cat = _feature_story_category(c["name"], ds)
        if cat not in stories:
            stories[cat] = {"total": 0, "items": []}
        stories[cat]["total"] += c["contribution"]
        stories[cat]["items"].append({"name": c["name"], "contribution": c["contribution"]})
    result = [{"category": k, "total": v["total"], "items": v["items"]} for k, v in stories.items()]
    result.sort(key=lambda x: abs(x["total"]), reverse=True)
    return {"prediction": pred, "stories": result}


@app.post("/api/counterfactual")
def counterfactual(data: dict):
    """Generate counterfactual 'what if' narratives."""
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "stories": []}
    stories = []
    # Top 3 positive contributors: "If you had less X, you'd save $Y"
    for c in [x for x in breakdown if x["contribution"] > 500][:3]:
        save = c["contribution"]
        pct = 100 * save / pred if pred else 0
        stories.append({
            "type": "reduce",
            "feature": c["name"],
            "impact": -save,
            "narrative": f"If this house had less {c['name'].replace('_', ' ').lower()}, you'd save about ${save:,.0f} ({pct:.1f}% of the price)."
        })
    # Top negative: "If you had more X, you'd pay $Y more"
    for c in [x for x in breakdown if x["contribution"] < -100][:2]:
        more = -c["contribution"]
        stories.append({
            "type": "increase",
            "feature": c["name"],
            "impact": more,
            "narrative": f"If this house had more {c['name'].replace('_', ' ').lower()}, you'd pay about ${more:,.0f} more."
        })
    return {"prediction": pred, "stories": stories[:5]}


@app.post("/api/sacrifice-options")
def sacrifice_options(data: dict):
    """Return trade-off options: 'To save $X, would you...'"""
    cal_defaults = {"total_rooms": 2635, "total_bedrooms": 537, "housing_median_age": 29, "median_income": 3.87, "population": 1425, "households": 499, "longitude": -119, "latitude": 36, "ocean_proximity": "INLAND"}
    hp_defaults = {"lot_area": 8450, "overall_qual": 7, "gr_liv_area": 1710, "garage_cars": 2, "total_bsmt_sf": 856, "year_built": 2003, "full_bath": 2, "fireplace": 0}
    defaults = cal_defaults if _is_hp_artifacts() else hp_defaults
    data = {**defaults, **{k: v for k, v in data.items() if v is not None}}
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "options": []}
    target_savings = float(data.get("target_savings", 50000))
    options = []
    ds = ARTIFACTS.get("dataset", "house_prices")
    # Map slider keys to human labels
    hp_labels = {
        "gr_liv_area": ("reduce living area by ~200 sq ft", 200),
        "overall_qual": ("lower quality by 1 point", 1),
        "lot_area": ("reduce lot size by ~1000 sq ft", 1000),
        "garage_cars": ("have 1 fewer garage space", 1),
        "total_bsmt_sf": ("reduce basement by ~100 sq ft", 100),
        "year_built": ("accept an older house (5 years)", 5),
        "full_bath": ("have 1 fewer bathroom", 1),
        "fireplace": ("have 1 fewer fireplace", 1),
    }
    cal_labels = {
        "total_rooms": ("reduce total rooms by ~200", 200),
        "total_bedrooms": ("reduce bedrooms by ~50", 50),
        "median_income": ("accept lower income area", 0.5),
        "housing_median_age": ("accept older housing (5 years)", 5),
        "population": ("move to less populated area", 200),
    }
    labels = cal_labels if _is_hp_artifacts() else hp_labels
    fe = ARTIFACTS.get("fe")
    model = ARTIFACTS["model"]
    for key, (label, delta) in labels.items():
        if _is_hp_artifacts():
            curr = float(data.get(key, 0))
            if key == "total_rooms" and curr > 400:
                new_val = max(2, curr - delta)
                new_data = {**data, key: new_val}
                p2, _, _ = _get_prediction_and_contributions(new_data)
                if p2 is not None and pred - p2 > 0:
                    options.append({"action": label, "savings": pred - p2, "new_value": new_val})
            elif key == "total_bedrooms" and curr > 100:
                new_val = max(1, curr - delta)
                new_data = {**data, key: new_val}
                p2, _, _ = _get_prediction_and_contributions(new_data)
                if p2 is not None and pred - p2 > 0:
                    options.append({"action": label, "savings": pred - p2, "new_value": new_val})
            elif key == "median_income" and curr > 1:
                new_val = max(0.5, curr - delta)
                new_data = {**data, key: new_val}
                p2, _, _ = _get_prediction_and_contributions(new_data)
                if p2 is not None and pred - p2 > 0:
                    options.append({"action": label, "savings": pred - p2, "new_value": new_val})
            elif key == "housing_median_age" and curr > 10:
                new_val = curr + delta
                new_data = {**data, key: new_val}
                p2, _, _ = _get_prediction_and_contributions(new_data)
                if p2 is not None and pred - p2 > 0:
                    options.append({"action": label, "savings": pred - p2, "new_value": new_val})
            elif key == "population" and curr > 500:
                new_val = max(3, curr - delta)
                new_data = {**data, key: new_val}
                p2, _, _ = _get_prediction_and_contributions(new_data)
                if p2 is not None and pred - p2 > 0:
                    options.append({"action": label, "savings": pred - p2, "new_value": new_val})
            continue
        if ds != "house_prices":
            continue
        curr = float(data.get(key, 0))
        if key == "gr_liv_area" and curr > 400:
            new_val = max(300, curr - 200)
            new_data = {**data, key: new_val}
            p2, _, _ = _get_prediction_and_contributions(new_data)
            if p2 is not None:
                save = pred - p2
                if save > 0:
                    options.append({"action": label, "savings": save, "new_value": new_val})
        elif key == "overall_qual" and curr > 1:
            new_data = {**data, key: curr - 1}
            p2, _, _ = _get_prediction_and_contributions(new_data)
            if p2 is not None:
                options.append({"action": label, "savings": pred - p2, "new_value": curr - 1})
        elif key == "lot_area" and curr > 1500:
            new_val = max(1000, curr - 1000)
            new_data = {**data, key: new_val}
            p2, _, _ = _get_prediction_and_contributions(new_data)
            if p2 is not None and pred - p2 > 0:
                options.append({"action": label, "savings": pred - p2, "new_value": new_val})
        elif key == "garage_cars" and curr > 0:
            new_data = {**data, key: curr - 1}
            p2, _, _ = _get_prediction_and_contributions(new_data)
            if p2 is not None and pred - p2 > 0:
                options.append({"action": label, "savings": pred - p2, "new_value": curr - 1})
    # Sort by savings closest to target
    options.sort(key=lambda o: abs(o["savings"] - target_savings))
    return {"prediction": pred, "target_savings": target_savings, "options": options[:5]}


@app.post("/api/sensitivity")
def sensitivity(data: dict):
    """Return cascade effects when one slider changes."""
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "cascades": []}
    keys = ["total_rooms", "median_income", "total_bedrooms", "housing_median_age"] if _is_hp_artifacts() else (["gr_liv_area", "overall_qual", "lot_area", "garage_cars"] if ARTIFACTS.get("dataset") == "house_prices" else [])
    cascades = []
    for key in keys:
        val = float(data.get(key, 0))
        if _is_hp_artifacts():
            step = 200 if "rooms" in key or "bedrooms" in key else (0.5 if "income" in key else 5)
        else:
            step = 200 if "area" in key or "liv" in key else 1
        new_data = {**data, key: val + step}
        p2, b2, _ = _get_prediction_and_contributions(new_data)
        if p2 is not None:
            cascades.append({
                "feature": key,
                "change": f"+{step}",
                "old_prediction": pred,
                "new_prediction": p2,
                "delta": p2 - pred,
            })
    return {"prediction": pred, "cascades": cascades}


@app.post("/api/dna-strand")
def dna_strand(data: dict):
    """Return strand data for Dream Home DNA viz (same as contributions, normalized for viz)."""
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "strands": []}
    total_abs = sum(abs(c["contribution"]) for c in breakdown) or 1
    strands = [{"name": c["name"], "contribution": c["contribution"], "segment": max(1, 100 * abs(c["contribution"]) / total_abs)} for c in breakdown[:12]]
    return {"prediction": pred, "strands": strands}


# Historical multipliers (synthetic): 2010=0.55, 2015=0.75, 2020=0.92, 2025=1.0
TIME_MULTIPLIERS = {2010: 0.55, 2012: 0.60, 2015: 0.75, 2018: 0.88, 2020: 0.92, 2022: 0.97, 2025: 1.0}


@app.post("/api/time-travel")
def time_travel(data: dict):
    """Return historical price estimates."""
    pred, _, _ = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "years": []}
    years = [{"year": y, "multiplier": m, "price": pred * m} for y, m in TIME_MULTIPLIERS.items()]
    return {"current_price": pred, "years": years}


def _climate_risk(lat: float, lon: float) -> dict:
    """Synthetic climate risk from lat/lon (California-focused)."""
    # Coastal = flood risk, inland south = fire risk
    flood = 0.3 + 0.4 * (1 - abs(lat - 34) / 10) if lon > -120 else 0.1
    fire = 0.2 + 0.5 * (lat - 33) / 5 if 33 < lat < 38 and lon > -119 else 0.1
    return {"flood_risk": min(1, max(0, flood)), "fire_risk": min(1, max(0, fire))}


@app.get("/api/climate-risk")
def climate_risk(lat: float = 36.0, lon: float = -119.0):
    """Return climate risk for a location."""
    r = _climate_risk(lat, lon)
    return {"latitude": lat, "longitude": lon, **r}


@app.get("/api/neighborhood-twins")
def neighborhood_twins(lat: float = 36.0, lon: float = -119.0):
    """Return similar neighborhoods (from map data)."""
    p = BASE / "data" / "housing.csv"
    if not p.exists():
        return {"twins": [], "message": "California housing data required."}
    df = pd.read_csv(p)
    if "longitude" not in df.columns:
        return {"twins": [], "message": "No location data."}
    df["dist"] = np.sqrt((df["longitude"] - lon) ** 2 + (df["latitude"] - lat) ** 2)
    df = df.sort_values("dist").head(20)
    # Group by rounded lat/lon for "neighborhoods"
    df["nb"] = (df["latitude"].round(1).astype(str) + "," + df["longitude"].round(1).astype(str))
    groups = df.groupby("nb").agg({"median_house_value": "median", "latitude": "first", "longitude": "first"}).reset_index()
    groups = groups.head(5)
    return {"twins": groups.to_dict(orient="records")}


@app.post("/api/confidence")
def confidence_landscape(data: dict):
    """Return confidence estimate (heuristic: distance from typical values)."""
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "confidence": 0.5}
    # Heuristic: more extreme contributions = lower confidence
    max_contrib = max(abs(c["contribution"]) for c in breakdown) if breakdown else 0
    conf = max(0.3, 1 - min(1, max_contrib / (pred * 0.5)))
    return {"prediction": pred, "confidence": round(conf, 2), "interpretation": "high" if conf > 0.7 else "medium" if conf > 0.5 else "low"}


@app.get("/api/fairness")
def fairness_lens():
    """Fairness lens - model uses only property features, no demographics."""
    return {
        "message": "This model uses only property and location features—no demographic data. It does not explicitly consider protected attributes.",
        "caveats": ["No demographic data in training.", "Location may correlate with demographics.", "Use for property comparison only."],
        "recommendation": "For fair lending, pair with human review and demographic fairness audits."
    }


@app.post("/api/simple-explain")
def simple_explain(data: dict):
    """Explainability for everyone - kid-friendly analogies."""
    pred, breakdown, fn = _get_prediction_and_contributions(data)
    if pred is None:
        return {"error": "Model not found.", "simple": ""}
    top = breakdown[0]
    name = top["name"].replace("_", " ").lower()
    contrib = top["contribution"]
    if "area" in name or "liv" in name or "lot" in name:
        analogy = f"This house costs more because it's like a bigger toy box—more space means a higher price!"
    elif "qual" in name or "quality" in name:
        analogy = f"This house costs more because it's like a shiny new toy—better quality means a higher price!"
    elif "bath" in name or "garage" in name:
        analogy = f"This house costs more because it has extra nice things—like more bathrooms or garage space!"
    else:
        analogy = f"The main reason for the price is {name}—it adds about ${abs(contrib):,.0f}."
    return {"prediction": pred, "simple": analogy, "top_feature": top["name"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
