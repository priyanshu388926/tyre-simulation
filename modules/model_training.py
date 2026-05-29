"""
model_training.py
-----------------
PRODUCTION SERVING MODULE — no training code lives here.

Responsibilities:
  • load_models()       – load pre-trained artifact files from disk
  • predict_row()       – run inference with the stacked ensemble
  • parameter_trend_forecast() – auto-predict sidebar utility

Training is done exclusively in train_models.py (run offline/locally).
"""

from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd

ARTIFACTS = "artifacts"
TARGETS   = ["mu_peak", "slip_peak", "mu_lock"]
BASE_KEYS = ["xgb", "rf", "lgb", "meta"]


# ---------------------------------------------------------------------------
# Artifact check helpers
# ---------------------------------------------------------------------------

def artifacts_exist(targets: list[str] = TARGETS,
                    out_dir: str = ARTIFACTS) -> tuple[bool, list[str]]:
    """Return (all_present, list_of_missing_paths)."""
    missing = []
    for t in targets:
        for key in BASE_KEYS:
            p = os.path.join(out_dir, f"{key}_{t}.joblib")
            if not os.path.exists(p):
                missing.append(p)
    feat_p = os.path.join(out_dir, "feature_cols.json")
    if not os.path.exists(feat_p):
        missing.append(feat_p)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Model loading  (use @st.cache_resource in the dashboard)
# ---------------------------------------------------------------------------

def load_models(targets: list[str] = TARGETS,
                out_dir: str = ARTIFACTS) -> dict:
    """
    Load stacked ensemble models for each target.

    Returns:
        {
          "mu_peak":   {"xgb": ..., "rf": ..., "lgb": ..., "meta": ...},
          "slip_peak": { ... },
          "mu_lock":   { ... },
        }
    """
    models = {}
    for t in targets:
        models[t] = {
            key: joblib.load(os.path.join(out_dir, f"{key}_{t}.joblib"))
            for key in BASE_KEYS
        }
    return models


def load_feature_cols(out_dir: str = ARTIFACTS) -> list[str]:
    with open(os.path.join(out_dir, "feature_cols.json")) as f:
        return json.load(f)


def load_report(out_dir: str = ARTIFACTS) -> dict:
    with open(os.path.join(out_dir, "report.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _base_predict(models_for_target: dict, X: np.ndarray) -> np.ndarray:
    """Get level-0 predictions from XGB, RF, LGB for a given input array."""
    X_f32 = X.astype(np.float32)
    xgb_pred  = models_for_target["xgb"].predict(X_f32)
    rf_pred   = models_for_target["rf"].predict(X_f32)
    # LightGBM trained on DataFrame — must receive float DataFrame
    lgb_pred  = models_for_target["lgb"].predict(
        pd.DataFrame(X_f32.astype(np.float64))
    )
    return np.column_stack([xgb_pred, rf_pred, lgb_pred]).astype(np.float32)


def predict_target(models_for_target: dict, X: np.ndarray) -> np.ndarray:
    """Stacked ensemble prediction for a single target (batch-safe)."""
    base = _base_predict(models_for_target, X)
    return models_for_target["meta"].predict(base.astype(np.float64))


def predict_all(all_models: dict,
                X_row: np.ndarray,
                feat_cols: list[str]) -> dict[str, float]:
    """
    Predict mu_peak, slip_peak, mu_lock for a single feature row.

    mu_lock uses mu_peak as an extra cross-target feature (appended column).
    X_row shape: (1, n_features)
    """
    mu_peak   = float(predict_target(all_models["mu_peak"],   X_row)[0])
    slip_peak = float(predict_target(all_models["slip_peak"], X_row)[0])

    # mu_lock: append mu_peak prediction as cross-target feature
    X_lock = np.hstack([X_row, [[mu_peak]]]).astype(np.float32)
    mu_lock = float(predict_target(all_models["mu_lock"], X_lock)[0])

    return {
        "mu_peak":   mu_peak,
        "slip_peak": max(min(slip_peak, 0.6), 0.02),
        "mu_lock":   mu_lock,
    }


# ---------------------------------------------------------------------------
# Trend forecaster (sidebar auto-predict)
# ---------------------------------------------------------------------------

def parameter_trend_forecast(series: pd.Series, n_ahead: int = 1) -> list[float]:
    """Blend linear-trend extrapolation with moving average for auto-predict."""
    s = pd.Series(series).dropna().astype(float).reset_index(drop=True)
    if len(s) < 3:
        return [float(s.iloc[-1])] * n_ahead if len(s) else [0.0] * n_ahead
    x = np.arange(len(s))
    slope, intercept = np.polyfit(x, s.values, 1)
    trend = [float(slope * (len(s) + i) + intercept) for i in range(n_ahead)]
    ma = float(s.tail(min(10, len(s))).mean())
    return [0.6 * t + 0.4 * ma for t in trend]
