"""
model_training.py
-----------------
Trains a weighted-average ensemble for µpeak, slip@µpeak and µlock.

Architecture per target
-----------------------
  Base learners (each tuned independently with RandomizedSearchCV):
    • XGBRegressor        – gradient-boosted trees
    • LGBMRegressor       – fast gradient boosting (if available)
    • RandomForestRegressor – bagged trees

  Ensemble strategy: weighted average where each model's weight is
  proportional to its 5-fold cross-validated R² on the training set.
  This is more robust than meta-learner stacking for small datasets
  (<500 rows) because it avoids the OOF/full-data distribution mismatch
  and removes the need for a second-level model to generalise.

Cross-target enrichment
-----------------------
  mu_lock is physically ≈ 0.7 × mu_peak. After predicting mu_peak we
  append its prediction as an extra feature when predicting mu_lock.
"""

from __future__ import annotations
import json
import os
import warnings
import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# ---------------------------------------------------------------------------
# Hyper-parameter search spaces
# ---------------------------------------------------------------------------
XGB_PARAM_SPACE = {
    "n_estimators": [300, 500, 700, 900, 1100],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_lambda": [0.5, 1.0, 2.0, 4.0],
    "reg_alpha": [0.0, 0.1, 0.3, 0.5],
    "gamma": [0.0, 0.05, 0.1, 0.3],
}

RF_PARAM_SPACE = {
    "n_estimators": [300, 500, 700],
    "max_depth": [None, 8, 12, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

LGBM_PARAM_SPACE = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [-1, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 0.3],
    "reg_lambda": [0.5, 1.0, 2.0],
    "min_child_samples": [5, 10, 20],
}


# ---------------------------------------------------------------------------
# Individual tuners – each returns a FITTED best estimator
# ---------------------------------------------------------------------------

def _tune_xgb(X: np.ndarray, y: np.ndarray,
              n_iter: int = 40, seed: int = 42) -> tuple[XGBRegressor, float, dict]:
    base = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=seed, n_jobs=-1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, XGB_PARAM_SPACE, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=seed, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    cv_r2 = float(search.best_score_)
    return search.best_estimator_, cv_r2, search.best_params_


def _tune_rf(X: np.ndarray, y: np.ndarray,
             n_iter: int = 20, seed: int = 42) -> tuple[RandomForestRegressor, float, dict]:
    base = RandomForestRegressor(random_state=seed, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, RF_PARAM_SPACE, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=seed, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    cv_r2 = float(search.best_score_)
    return search.best_estimator_, cv_r2, search.best_params_


def _tune_lgbm(X: np.ndarray, y: np.ndarray,
               n_iter: int = 30, seed: int = 42):
    # Convert to DataFrame so LightGBM stores feature names consistently.
    X_df = pd.DataFrame(X)
    base = LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, LGBM_PARAM_SPACE, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=seed, n_jobs=-1, verbose=0,
    )
    search.fit(X_df, y)
    cv_r2 = float(search.best_score_)
    return search.best_estimator_, cv_r2, search.best_params_


# ---------------------------------------------------------------------------
# Weighted-average ensemble wrapper
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """Predict as a weighted average of base estimators.

    Weights are the CV R² scores from tuning; any negative R² is clipped to 0
    so bad models don't drag predictions backwards. Weights are normalised to
    sum to 1.
    """

    def __init__(self, estimators: list, weights: list[float]):
        """
        estimators : list of (name, fitted_model, is_lgbm) tuples
        weights    : raw CV-R² weights (will be normalised internally)
        """
        self.estimators = estimators
        raw = np.clip(np.array(weights, dtype=float), 0, None)
        total = raw.sum()
        self.weights_ = (raw / total) if total > 0 else np.ones(len(raw)) / len(raw)
        self.named_estimators_ = {name: est for name, est, _ in estimators}

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for (name, est, is_lgbm) in self.estimators:
            if is_lgbm:
                X_in = pd.DataFrame(X.astype(float))   # LightGBM requires float, not object
            else:
                X_in = X.astype(float)
            preds.append(est.predict(X_in))
        return np.average(np.column_stack(preds), axis=1, weights=self.weights_)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_all(X: pd.DataFrame, Y: pd.DataFrame, out_dir: str = "artifacts",
              n_iter: int = 40) -> dict:
    """Train one weighted ensemble per target column in Y. Persist models."""
    os.makedirs(out_dir, exist_ok=True)
    report = {"targets": {}, "features": list(X.columns), "lgbm_available": _HAS_LGBM}

    X_vals = X.values
    oof_mu_peak: np.ndarray | None = None

    TARGET_ORDER = ["mu_peak", "slip_peak", "mu_lock"]
    ordered_cols = [t for t in TARGET_ORDER if t in Y.columns] + \
                   [t for t in Y.columns if t not in TARGET_ORDER]

    for target in ordered_cols:
        y = Y[target].values

        # Cross-target enrichment: append mu_peak prediction for mu_lock.
        X_use = X_vals
        if target == "mu_lock" and oof_mu_peak is not None:
            X_use = np.hstack([X_vals, oof_mu_peak.reshape(-1, 1)])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_use, y, test_size=0.2, random_state=42
        )

        # Tune each base model on the training set.
        xgb_est, xgb_r2, xgb_params = _tune_xgb(X_tr, y_tr, n_iter=n_iter, seed=42)
        rf_est,  rf_r2,  rf_params  = _tune_rf(X_tr, y_tr,
                                                n_iter=max(10, n_iter // 2), seed=42)

        estimators = [("xgb", xgb_est, False), ("rf", rf_est, False)]
        weights    = [xgb_r2, rf_r2]
        all_params = {"xgb": xgb_params, "rf": rf_params}

        if _HAS_LGBM:
            lgbm_est, lgbm_r2, lgbm_params = _tune_lgbm(
                X_tr, y_tr, n_iter=max(15, n_iter // 2), seed=42
            )
            estimators.append(("lgbm", lgbm_est, True))
            weights.append(lgbm_r2)
            all_params["lgbm"] = lgbm_params

        ensemble = WeightedEnsemble(estimators=estimators, weights=weights)

        pred = ensemble.predict(X_te)
        r2   = float(r2_score(y_te, pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))

        # Store XGB predictions on full X for cross-target enrichment.
        if target == "mu_peak":
            oof_mu_peak = ensemble.predict(X_vals)

        # Feature importance from XGB base.
        try:
            fi = dict(zip(X.columns,
                          xgb_est.feature_importances_[:len(X.columns)].tolist()))
        except Exception:
            fi = {}

        w_info = {name: float(w) for (name, _, __), w in zip(estimators, ensemble.weights_)}
        joblib.dump(ensemble, os.path.join(out_dir, f"xgb_{target}.joblib"))
        report["targets"][target] = {
            "r2": r2, "rmse": rmse,
            "ensemble_weights": w_info,
            "cv_r2": {"xgb": xgb_r2, "rf": rf_r2,
                      **( {"lgbm": lgbm_r2} if _HAS_LGBM else {})},
            "best_params": all_params,
            "feature_importance": fi,
        }
        badge = "✅" if r2 >= 0.85 else ("⚠️" if r2 >= 0.70 else "❌")
        print(f"  {badge} [{target}]  R²={r2:.4f}  RMSE={rmse:.5f}"
              f"  weights={w_info}")

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def load_models(targets: list[str], out_dir: str = "artifacts") -> dict:
    return {t: joblib.load(os.path.join(out_dir, f"xgb_{t}.joblib")) for t in targets}


def predict_row(models: dict, feature_row: pd.DataFrame,
                mu_peak_val: float | None = None) -> dict:
    """Predict all targets for a single-row aligned feature DataFrame.

    mu_lock was trained with mu_peak appended as a cross-target feature (n+1
    columns). Pass mu_peak_val to include it; omit to skip mu_lock entirely so
    the caller can make a second pass with the known mu_peak value.
    """
    results = {}
    for t, m in models.items():
        X_in = feature_row.values
        if t == "mu_lock":
            if mu_peak_val is None:
                continue          # skip — requires mu_peak cross-target feature
            X_in = np.hstack([X_in, [[mu_peak_val]]])
        results[t] = float(m.predict(X_in)[0])
    return results


def parameter_trend_forecast(series: pd.Series, n_ahead: int = 1) -> list[float]:
    """Simple but robust 1D forecaster for a parameter trend."""
    s = pd.Series(series).dropna().astype(float).reset_index(drop=True)
    if len(s) < 3:
        return [float(s.iloc[-1])] * n_ahead if len(s) else [0.0] * n_ahead
    x = np.arange(len(s))
    slope, intercept = np.polyfit(x, s.values, 1)
    trend_pred = [float(slope * (len(s) + i) + intercept) for i in range(n_ahead)]
    ma = float(s.tail(min(10, len(s))).mean())
    blended = [0.6 * t + 0.4 * ma for t in trend_pred]
    return blended
