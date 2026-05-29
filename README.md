# Tyre Traction Prediction & Simulation System

End-to-end ML + physics platform for tyre traction analysis. Predicts
**µpeak**, **slip @ µpeak** and **µlock** from tyre + test parameters,
then synthesises a **Pacejka Magic-Formula** µ–slip curve and serves it
through a Streamlit engineering dashboard.

## Project structure
```
tyre_sim/
├── dashboard_app.py          # Streamlit UI
├── requirements.txt
├── data/
│   └── apollo.xlsx           # your dataset (already included)
└── modules/
    ├── data_processing.py    # ingestion, KNN imputation, physics features
    ├── model_training.py     # XGBoost training + tuning + trend forecast
    ├── curve_model.py        # Gaussian + Pacejka curve generators
    └── report_generator.py   # PDF export
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run dashboard_app.py
```
On first launch the app trains XGBoost models (CV + RandomizedSearch) and
caches them under `artifacts/`. Click **Retrain models** in the sidebar
to refresh.

## How it works
1. **Data pipeline** (`data_processing.py`)
   - Fuzzy-matches messy column headers to a canonical schema.
   - KNN-imputes missing numeric values (preserves correlations between
     correlated tyre parameters such as load/pressure).
   - Injects physics features: sidewall height, load-per-width,
     pressure/load ratio, speed², aspect × pressure.
   - One-hot encodes brand / pattern / compound / group / surface.
2. **ML models** (`model_training.py`)
   - One `XGBRegressor` per target with `RandomizedSearchCV` over depth,
     learning rate, subsampling and regularisation, scored by 5-fold R².
   - Persists models + a JSON report with R², RMSE and feature importance.
3. **Trend forecaster**
   - `parameter_trend_forecast` blends linear-fit extrapolation with a
     moving average so the "Auto-predict" sidebar mode proposes the next
     plausible operating point from historical tests.
4. **Curve synthesis** (`curve_model.py`)
   - Baseline Gaussian centred on `slip_peak`.
   - Pacejka Magic Formula with B,C,D,E estimated from ML anchors
     (`D = µpeak`, `B = π/(2·C·slip_peak)`, `E` derived from µlock/µpeak).
   - Physics > ML alone: Pacejka captures the asymmetric peak and the
     post-lock plateau that a pure regressor cannot extrapolate.
5. **Dashboard** (`dashboard_app.py`)
   - Sidebar: manual inputs or auto-predicted inputs.
   - Tabs: live µ–slip curve, animated build, two-tyre comparison,
     top-20 feature importance, raw dataset preview.
   - Right panel: predictions, Pacejka coefficients, model hold-out
     metrics, auto-generated engineering insight, PDF export.
