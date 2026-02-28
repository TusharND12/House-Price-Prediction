# SmartExplain AI – Interpretable & Adaptive House Price Prediction Engine

Production-level ML project combining research-style Jupyter notebook, modular Python architecture, and reproducible pipeline for California house price prediction.

## Project Overview

SmartExplain AI implements linear regression from scratch using gradient descent (batch, mini-batch, SGD) with:
- L2 regularization
- Momentum and learning rate decay
- Early stopping
- Feature contribution explainability
- What-if simulation
- Interactive Streamlit app

**Main model:** Custom `LinearRegressionGD` (sklearn LinearRegression used only for comparison).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SmartExplain AI                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  data/housing.csv                                                       │
│         │                                                               │
│         ▼                                                               │
│  core/feature_engineering.py  ──►  X, y (standardized + engineered)     │
│         │                                                               │
│         ▼                                                               │
│  core/model.py (LinearRegressionGD)                                     │
│    - Batch / Mini-batch / SGD                                           │
│    - core/optimizers.py (Momentum, LR decay)                            │
│         │                                                               │
│         ▼                                                               │
│  core/explainability.py  ──►  Contribution_i = w_i * x_i                │
│         │                                                               │
│         ├──► simulator/what_if.py  (simulate_price_change)              │
│         │                                                               │
│         ├──► visualization/plots.py  (cost, actual vs pred)             │
│         ├──► visualization/cost_surface.py  (3D cost surface)           │
│         │                                                               │
│         └──► app/streamlit_app.py  (interactive UI)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Background

**Linear model:**  
$$y = Xw + b$$

**Cost (L2 regularized MSE):**  
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (y_{pred}^{(i)} - y^{(i)})^2 + \lambda \sum_{j} w_j^2$$

**Gradients:**  
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (y_{pred} - y) + 2\lambda w$$  
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i} (y_{pred}^{(i)} - y^{(i)})$$

**Feature contribution:**  
$$\text{Contribution}_i = w_i \cdot x_i$$

---

## How to Run

### 1. Install dependencies

```bash
cd SmartExplain-AI
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook notebooks/SmartExplain_AI.ipynb
# or
jupyter lab notebooks/SmartExplain_AI.ipynb
```

### 3. Train the model

```bash
python train.py
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
SmartExplain-AI/
├── data/
│   └── housing.csv
├── notebooks/
│   └── SmartExplain_AI.ipynb
├── core/
│   ├── model.py           # LinearRegressionGD
│   ├── optimizers.py      # Momentum, LR decay
│   ├── feature_engineering.py
│   ├── metrics.py         # MAE, MSE, RMSE, R² (manual)
│   └── explainability.py
├── visualization/
│   ├── plots.py
│   └── cost_surface.py
├── simulator/
│   └── what_if.py
├── app/
│   └── streamlit_app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Results Summary

Typical metrics on the full California housing dataset:
- **MAE:** ~50,000–60,000
- **RMSE:** ~70,000–80,000
- **R²:** ~0.65–0.70 (similar to sklearn LinearRegression)

---

## Future Improvements

- Train/validation split and cross-validation
- Neural networks or tree-based models
- Uncertainty quantification
- Hyperparameter tuning (learning rate, regularization)
- More sophisticated feature engineering
