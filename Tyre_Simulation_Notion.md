# 📋 Tyre Traction Prediction & Simulation System: Objectives & Implementation Details

## 1) Data Ingestion & Standardization
- **Excel to CSV Ingestion**: Handled in `modules/data_processing.py` via `load_dataset`. It automatically reads raw data from `data/apollo.xlsx` and exports a fast-loading, flat `data/apollo.csv` flat file during offline training in `train_models.py`.
- **Fuzzy Column Matching**: Standardizes messy sheet headers using a canonical alias dictionary `CANONICAL` and a two-pass substring matcher implemented in `detect_columns`.
- **Data Typing & Casting**: Isolates date columns, and casts all numeric fields to memory-efficient `float32` arrays in the `preprocess` pipeline to optimize dashboard load times.
- **Friction Character Encoding**: Normalizes Unicode symbols (standardizing `μ` → `µ`) inside the `_norm` utility to ensure dictionary lookups never fail.

## 2) Data Quality Checks
- **Outlier Pruning**: Prunes outliers prior to imputation using a 4.0σ Z-score threshold in `remove_outliers` to avoid skewing neighbor weights during KNN calculation.
- **Unit Consistency**: Validates that all physical properties align with canonical units: Load (N), Inflation Pressure (kPa), and Speed (km/h).
- **Duplicate Handling**: 
  > ⚠️ **Important Action Item**: Explicit `.drop_duplicates()` filtering based on a composite key of `(Project + Tyre + Test Condition + Trial No)` needs to be added to the preprocessing pipeline to remove redundant runs.

## 3) Feature Engineering
- **Standard Tyre Dimensions**: Extracts and utilizes Width, Aspect Ratio, Rim Diameter, and Rim Width directly.
- **Physics-Informed Features**: Derived in `add_physics_features` inside `modules/data_processing.py` to enrich modeling performance. This includes 12 custom mechanical variables such as:
  - `sidewall_height_mm` = (Aspect Ratio × Width) / 100
  - `contact_patch_area` = Width × (Load / Pressure)
  - `load_per_width` = Load / Width
  - `pressure_load_ratio` = Pressure / Load
  - `speed_sq` = Speed²
- **Categorical Encodings**: Transforms Brand, Pattern, Compound, and Surface/Test type features into one-hot encodings via `pd.get_dummies` with `dtype=int` to bypass the pandas 2.x boolean formatting issue.
- **Slip Ratio Scale Normalization**: Normalizes raw percentage slip inputs (e.g., 13.8%) to decimal fractions (e.g., 0.138) for modeling compatibility.

## 4) Modeling Goals
- **Ensemble Target Predictors**: Orchestrates prediction via stacked regression models trained in `train_models.py` and served via `modules/model_training.py` for:
  1. `mu_peak` (primary friction peak target)
  2. `slip_peak` (optimal slip ratio)
  3. `mu_lock` (sliding friction coefficient at 100% lockup)
- **Unified Model with Surface Flags**: Utilizes surface attributes (`test_type` flags) as active model features instead of training isolated models per surface, preserving the maximum possible sample size.
- **Cross-Target Prediction Pattern**: Respects vehicle physics by appending the out-of-fold (OOF) predictions of the `mu_peak` model as an extra input feature when training the level-0 and level-1 ensembles for `mu_lock`. This allows the model to exploit the physical correlation between peak and sliding friction without data leakage.

## 5) Train/Validation Strategy
- **Multi-Model Stacking**: Trains XGBoost, Random Forest, and LightGBM base models (Level 0) and blends their out-of-fold predictions using a Ridge Regression meta-learner (Level 1) in `train_models.py`.

## 6) Deliverables
- **Cleaned Dataset**: Preprocessed, standardized dataset cached under `data/apollo.csv`.
- **Artifact Packages**: Persisted scaling estimators, trained level-0/level-1 models, and feature maps saved as `.joblib` and `.json` files under the `artifacts/` folder.
- **Inference Template**: Implemented in `predict_all` inside `modules/model_training.py` to manage step-dependent inferences.
- **Interactive Dashboard**: Served in `dashboard_app.py`. Contains interactive inputs, Pacejka curve synthesis, two-tyre comparisons, feature importance visualizers, and one-click PDF generation.

---

## 🛞 Pacejka Magic Formula Curve Synthesis

- **Purpose**: Reconstructs a physically realistic, continuous friction-vs-slip curve from the discrete targets (`mu_peak`, `slip_peak`, `mu_lock`) predicted by the machine learning models.
- **Equation**: 
  $$\mu(s) = D \sin\Big(C \arctan\big(B s - E (B s - \arctan(B s))\big)\Big)$$
- **Peak Factor (D)**: Dictates the maximum height of the curve, set to the ML-predicted `mu_peak`.
- **Shape Factor (C)**: Determines the limits of the sine function. Set to the standard longitudinal constant of 1.65.
- **Stiffness Factor (B)**: Controls the slope at zero slip. Derived algebraically as:
  $$B = \frac{\pi}{2 C \cdot \text{slip\_peak}}$$
  to guarantee the curve peaks exactly at the predicted `slip_peak`.
- **Curvature Factor (E)**: Models the post-peak drop-off. Estimated from the ratio of slide friction to peak friction:
  $$E = \text{clip}\left(1.0 - \frac{\text{mu\_lock}}{\text{mu\_peak}}, -2.0, 0.98\right)$$
- **Implementation Location**: Implemented in `pacejka_curve` and `estimate_pacejka_params` within `modules/curve_model.py`.

---

## 📂 File Architecture & Purpose

### Main Project Files
1. **`dashboard_app.py`**
   - **Purpose**: Streamlit application rendering the visual dashboard interface. Orchestrates input collection, side-by-side comparison plotting, feature importance graphs, animations, and PDF report creation. It operates entirely as an inference engine (never trains models or runs searches on start) for instant cold starts.
2. **`train_models.py`**
   - **Purpose**: The offline training orchestration script. Reads the Excel dataset, cleans it, splits train/test partitions, executes hyperparameter tuning (`RandomizedSearchCV`), generates OOF predictions to train the stacked meta-learner (Ridge Regression), and dumps all joblib artifacts, reports, and fast-load CSVs.
3. **`requirements.txt`**
   - **Purpose**: Defines dependencies for core ML (scikit-learn, xgboost, lightgbm, joblib), data processing (pandas, openpyxl), and the dashboard/PDF reporting utility (streamlit, plotly, reportlab, matplotlib).
4. **`render.yaml`**
   - **Purpose**: Configuration instructions for deploying the dashboard application on hosting tiers like Render.

### Core Modules (`modules/` folder)
5. **`modules/data_processing.py`**
   - **Purpose**: Encapsulates raw data preprocessing. Standardizes text and column names, runs Z-score outlier detection, executes physics feature engineering, and performs adaptive KNN-imputation on numeric properties.
6. **`modules/model_training.py`**
   - **Purpose**: Production serving module. Loads saved model estimators, formats live feature rows to align with training schemas, makes ensemble predictions (with cross-target enrichment), and houses trend forecasting logic for manual/auto-prediction inputs.
7. **`modules/curve_model.py`**
   - **Purpose**: Manages the mathematical curve synthesis. Contains calculations for Gaussian baseline curves and Pacejka longitudinal formulas using the ML anchors, and generates natural-language engineering commentary.
8. **`modules/report_generator.py`**
   - **Purpose**: Generates dynamic, single-page PDF reports including input parameter grids, predicted targets, Pacejka constants, visual curve comparisons plotted with Matplotlib, and engineering notes.
