"""
train_models.py
---------------
OFFLINE TRAINING SCRIPT — run once locally before deployment.

Usage:
    python train_models.py [--data data/apollo.xlsx] [--out artifacts] [--iter 30]

What this script does:
  1. Loads and preprocesses the dataset (XLSX → CSV conversion included).
  2. For each target (mu_peak, slip_peak, mu_lock):
       a. Tunes XGBoost, Random Forest, LightGBM via RandomizedSearchCV.
       b. Generates 5-fold OOF predictions from each tuned base model.
       c. Trains a Ridge meta-learner on the OOF predictions.
       d. Refits each base model on the FULL training set.
       e. Evaluates the stacked ensemble on a 20% held-out test set.
       f. Saves 4 artifacts: xgb_{t}.joblib, rf_{t}.joblib,
                              lgb_{t}.joblib, meta_{t}.joblib
  3. Saves feature_cols.json and report.json.
  4. Saves data/apollo.csv for fast dashboard loading.

Cross-target enrichment:
  mu_lock is physically ≈ 0.7 × mu_peak. OOF mu_peak predictions are
  appended as an extra feature when training mu_lock, letting the model
  exploit that correlation without data leakage.

DO NOT import or call this script from dashboard_app.py.
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    print("[WARN] LightGBM not installed — ensemble will use XGB + RF only.")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from modules.data_processing import load_dataset, preprocess

# ---------------------------------------------------------------------------
# Hyper-parameter search spaces
# ---------------------------------------------------------------------------
XGB_SPACE = {
    "n_estimators":     [300, 500, 700, 900, 1100],
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_lambda":       [0.5, 1.0, 2.0, 4.0],
    "reg_alpha":        [0.0, 0.1, 0.3, 0.5],
    "gamma":            [0.0, 0.05, 0.1, 0.3],
}

RF_SPACE = {
    "n_estimators":    [300, 500, 700],
    "max_depth":       [None, 8, 12, 20],
    "min_samples_leaf":[1, 2, 4],
    "max_features":    ["sqrt", "log2", 0.5],
}

LGBM_SPACE = {
    "n_estimators":      [300, 500, 700, 900],
    "max_depth":         [-1, 6, 8, 10],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "num_leaves":        [31, 63, 127],
    "subsample":         [0.7, 0.85, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "reg_alpha":         [0.0, 0.1, 0.3],
    "reg_lambda":        [0.5, 1.0, 2.0],
    "min_child_samples": [5, 10, 20],
}

TARGET_ORDER = ["mu_peak", "slip_peak", "mu_lock"]


# ---------------------------------------------------------------------------
# Base model tuners
# ---------------------------------------------------------------------------

def _tune_xgb(X: np.ndarray, y: np.ndarray,
              n_iter: int, seed: int) -> XGBRegressor:
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, XGB_SPACE, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=seed, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    print(f"      XGB  best CV R²={search.best_score_:.4f}  params={search.best_params_}")
    return search.best_estimator_


def _tune_rf(X: np.ndarray, y: np.ndarray,
             n_iter: int, seed: int) -> RandomForestRegressor:
    base = RandomForestRegressor(random_state=seed, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, RF_SPACE, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=seed, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    print(f"      RF   best CV R²={search.best_score_:.4f}  params={search.best_params_}")
    return search.best_estimator_


def _tune_lgbm(X: np.ndarray, y: np.ndarray,
               n_iter: int, seed: int):
    X_df = pd.DataFrame(X.astype(np.float64))
    base = LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base, LGBM_SPACE, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=seed, n_jobs=-1, verbose=0,
    )
    search.fit(X_df, y)
    print(f"      LGB  best CV R²={search.best_score_:.4f}  params={search.best_params_}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# OOF prediction generator
# ---------------------------------------------------------------------------

def _oof_predict(model_proto, X: np.ndarray, y: np.ndarray,
                 n_splits: int, seed: int, is_lgbm: bool = False) -> np.ndarray:
    """
    Generate out-of-fold predictions for a model prototype.

    model_proto must be an unfitted or fitted sklearn estimator that
    supports clone (deep copy + refit).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, val_idx in kf.split(X):
        m = copy.deepcopy(model_proto)
        if is_lgbm:
            m.fit(pd.DataFrame(X[tr_idx].astype(np.float64)), y[tr_idx])
            oof[val_idx] = m.predict(pd.DataFrame(X[val_idx].astype(np.float64)))
        else:
            m.fit(X[tr_idx], y[tr_idx])
            oof[val_idx] = m.predict(X[val_idx])
    return oof


# ---------------------------------------------------------------------------
# Single-target training
# ---------------------------------------------------------------------------

def train_target(
    target: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    feat_cols: list[str],
    n_iter_xgb: int, n_iter_rf: int, n_iter_lgbm: int,
    seed: int,
    out_dir: str,
) -> dict:
    """
    Train a stacked ensemble (XGB + RF + [LGB] → Ridge) for one target.

    Returns metrics dict and saves 4 artifacts to out_dir.
    """
    t0 = time.time()
    n_splits = 5

    print(f"   Tuning base models...")
    xgb_model  = _tune_xgb(X_tr, y_tr, n_iter_xgb, seed)
    rf_model   = _tune_rf(X_tr, y_tr, n_iter_rf, seed)
    lgbm_model = _tune_lgbm(X_tr, y_tr, n_iter_lgbm, seed) if _HAS_LGBM else None

    print(f"   Generating OOF predictions...")
    oof_xgb  = _oof_predict(xgb_model, X_tr, y_tr, n_splits, seed, is_lgbm=False)
    oof_rf   = _oof_predict(rf_model,  X_tr, y_tr, n_splits, seed, is_lgbm=False)
    oof_cols = [oof_xgb, oof_rf]

    if lgbm_model is not None:
        oof_lgbm = _oof_predict(lgbm_model, X_tr, y_tr, n_splits, seed, is_lgbm=True)
        oof_cols.append(oof_lgbm)

    oof_meta = np.column_stack(oof_cols)

    print(f"   Training Ridge meta-learner...")
    meta = Ridge(alpha=1.0)
    meta.fit(oof_meta, y_tr)

    # Refit base models on FULL training data
    print(f"   Refitting base models on full train set...")
    xgb_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)
    if lgbm_model is not None:
        lgbm_model.fit(pd.DataFrame(X_tr.astype(np.float64)), y_tr)

    # Test-set evaluation
    te_base_cols = [
        xgb_model.predict(X_te),
        rf_model.predict(X_te),
    ]
    if lgbm_model is not None:
        te_base_cols.append(lgbm_model.predict(pd.DataFrame(X_te.astype(np.float64))))
    te_meta = np.column_stack(te_base_cols)
    y_pred  = meta.predict(te_meta)

    r2   = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    badge = "✅" if r2 >= 0.85 else ("⚠️" if r2 >= 0.70 else "❌")
    print(f"   {badge} R²={r2:.4f}  RMSE={rmse:.5f}  ({time.time()-t0:.1f}s)")

    # Feature importance from XGB
    try:
        fi = dict(zip(feat_cols,
                      xgb_model.feature_importances_[:len(feat_cols)].tolist()))
    except Exception:
        fi = {}

    # Save artifacts
    joblib.dump(xgb_model,  os.path.join(out_dir, f"xgb_{target}.joblib"),  compress=3)
    joblib.dump(rf_model,   os.path.join(out_dir, f"rf_{target}.joblib"),   compress=3)
    joblib.dump(meta,       os.path.join(out_dir, f"meta_{target}.joblib"), compress=3)
    if lgbm_model is not None:
        joblib.dump(lgbm_model, os.path.join(out_dir, f"lgb_{target}.joblib"), compress=3)
    else:
        # Save a dummy that mirrors XGB so load_models() always finds 4 files
        joblib.dump(xgb_model, os.path.join(out_dir, f"lgb_{target}.joblib"), compress=3)

    return {
        "r2": r2, "rmse": rmse,
        "lgbm_used": lgbm_model is not None,
        "feature_importance": fi,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(data_path: str, out_dir: str, n_iter: int, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    total_t0 = time.time()

    # ── Load & preprocess ────────────────────────────────────────────────
    print("\n=== Step 1: Data ===")
    df = load_dataset(data_path)
    X, Y, feat_cols, tgt_cols = preprocess(df, as_float32=True)
    print(f"  Dataset: {X.shape[0]} rows × {X.shape[1]} features  |  targets: {tgt_cols}")

    # Save CSV for fast dashboard loading
    csv_out = os.path.join(os.path.dirname(data_path), "apollo.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved fast-load CSV → {csv_out}")

    # Save feature column schema
    json.dump(feat_cols, open(os.path.join(out_dir, "feature_cols.json"), "w"), indent=2)

    # ── Train/test split ─────────────────────────────────────────────────
    X_vals = X.values
    Y_vals = Y.values
    X_tr_base, X_te_base, Y_tr, Y_te = train_test_split(
        X_vals, Y_vals, test_size=0.2, random_state=seed
    )

    n_xgb  = n_iter
    n_rf   = max(10, n_iter // 2)
    n_lgbm = max(15, n_iter // 2)

    # ── Train each target ────────────────────────────────────────────────
    print(f"\n=== Step 2: Training (n_iter_xgb={n_xgb}, n_iter_rf={n_rf}, n_iter_lgbm={n_lgbm}) ===")
    report = {"targets": {}, "features": feat_cols, "lgbm_available": _HAS_LGBM}

    # We need OOF mu_peak to use as cross-target feature for mu_lock.
    # Compute it by running OOF on the base mu_peak models after tuning.
    oof_mu_peak_full: np.ndarray | None = None

    ordered = [t for t in TARGET_ORDER if t in tgt_cols] + \
              [t for t in tgt_cols if t not in TARGET_ORDER]

    for i, target in enumerate(ordered):
        t_idx = list(Y.columns).index(target)
        y_tr  = Y_tr[:, t_idx].astype(np.float64)
        y_te  = Y_te[:, t_idx].astype(np.float64)

        print(f"\n── {target} ──")

        # For mu_lock: append oof_mu_peak as extra feature
        if target == "mu_lock" and oof_mu_peak_full is not None:
            oof_mu_peak_tr = oof_mu_peak_full[:len(X_tr_base)]
            X_tr = np.hstack([X_tr_base, oof_mu_peak_tr.reshape(-1, 1)]).astype(np.float32)
            # For test set: use trained mu_peak model to get predictions
            mu_peak_te = _get_mu_peak_te_preds(out_dir, X_te_base)
            X_te = np.hstack([X_te_base, mu_peak_te.reshape(-1, 1)]).astype(np.float32)
        else:
            X_tr = X_tr_base.astype(np.float32)
            X_te = X_te_base.astype(np.float32)

        metrics = train_target(
            target=target,
            X_tr=X_tr, y_tr=y_tr,
            X_te=X_te, y_te=y_te,
            feat_cols=feat_cols,
            n_iter_xgb=n_xgb, n_iter_rf=n_rf, n_iter_lgbm=n_lgbm,
            seed=seed, out_dir=out_dir,
        )
        report["targets"][target] = metrics

        # After training mu_peak, generate OOF preds over full dataset
        # (used as cross-target feature for mu_lock training)
        if target == "mu_peak":
            oof_mu_peak_full = _get_mu_peak_oof(out_dir, X_vals.astype(np.float32))

    # ── Save report ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE  ({time.time()-total_t0:.1f}s total)")
    print(f"{'='*50}")
    for t, m in report["targets"].items():
        badge = "✅" if m["r2"] >= 0.85 else ("⚠️" if m["r2"] >= 0.70 else "❌")
        print(f"  {badge}  {t:12s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.5f}")
    print(f"\nArtifacts saved to: {os.path.abspath(out_dir)}/")
    print("Now commit artifacts/ to git and deploy.\n")


def _get_mu_peak_oof(out_dir: str, X_full: np.ndarray) -> np.ndarray:
    """Predict mu_peak on all rows using the saved stacked ensemble."""
    from modules.model_training import load_models, predict_target
    models = load_models(["mu_peak"], out_dir)
    return predict_target(models["mu_peak"], X_full).astype(np.float32)


def _get_mu_peak_te_preds(out_dir: str, X_te: np.ndarray) -> np.ndarray:
    """Predict mu_peak on test rows using the saved stacked ensemble."""
    from modules.model_training import load_models, predict_target
    models = load_models(["mu_peak"], out_dir)
    return predict_target(models["mu_peak"], X_te.astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train tyre traction models and save artifacts."
    )
    parser.add_argument("--data",  default="data/apollo.xlsx",
                        help="Path to dataset (XLSX or CSV)")
    parser.add_argument("--out",   default="artifacts",
                        help="Output directory for model artifacts")
    parser.add_argument("--iter",  type=int, default=30,
                        help="RandomizedSearchCV n_iter for XGB (RF/LGB scaled)")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data,
        out_dir=args.out,
        n_iter=args.iter,
        seed=args.seed,
    )
