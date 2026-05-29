"""
data_processing.py
------------------
Ingestion, feature engineering, and preprocessing.
Production rules:
  - No model fitting here.
  - All numeric output is float32 to reduce memory footprint.
  - The pandas 2.x get_dummies bool-dtype bug is fixed (dtype=int).
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Canonical column schema
# ---------------------------------------------------------------------------
CANONICAL = {
    "width":        ["sw", "section width", "width", "sw mm", "sw,mm"],
    "aspect_ratio": ["ar", "aspect ratio", "aspect"],
    "rim_dia":      ["rim dia", "rim diameter", "rim dia inch", "rim"],
    "rim_width":    ["rim width", "rim width inch"],
    "load":         ["test load", "load", "fz", "vertical load"],
    "pressure":     ["test ip", "ip", "pressure", "inflation pressure"],
    "speed":        ["test speed", "speed", "vx"],
    "brand":        ["brand", "manufacturer"],
    "pattern":      ["pattern", "tread pattern"],
    "compound":     ["tread cmpd", "compound", "tread compound"],
    "group":        ["group"],
    "test_type":    ["test details", "test type", "surface"],
    "trial":        ["trial", "trial no"],
    "mu_peak":      ["µpeak", "mu peak", "mupeak", "peak mu", "peak friction"],
    "slip_peak":    ["sr @ µpeak", "slip peak", "sr peak", "slip at peak"],
    "mu_lock":      ["µlock", "mu lock", "mulock", "lock mu"],
    "date":         ["test date", "date"],
}

TARGETS = ("mu_peak", "slip_peak", "mu_lock")
CAT_COLS = ("brand", "pattern", "compound", "group", "test_type")


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("\n", " ").replace("μ", "µ")
    s = re.sub(r"[^a-z0-9µ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Two-pass fuzzy column matching: exact first, then substring."""
    raw = {c: _norm(c) for c in df.columns}
    used: set[str] = set()
    mapping: dict[str, str] = {}

    for canon, aliases in CANONICAL.items():
        norm_aliases = [_norm(a) for a in aliases]
        for raw_col, norm in raw.items():
            if raw_col in used:
                continue
            if norm in norm_aliases:
                mapping[canon] = raw_col
                used.add(raw_col)
                break

    for canon, aliases in CANONICAL.items():
        if canon in mapping:
            continue
        norm_aliases = [_norm(a) for a in aliases]
        for raw_col, norm in raw.items():
            if raw_col in used:
                continue
            if any(a and (a in norm or norm in a) for a in norm_aliases):
                mapping[canon] = raw_col
                used.add(raw_col)
                break

    return mapping


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str, sheet: str | int = 0) -> pd.DataFrame:
    """Load Excel and rename to canonical schema."""
    df = pd.read_excel(path, sheet_name=sheet)
    mapping = detect_columns(df)
    df = df.rename(columns={v: k for k, v in mapping.items()})
    keep = [c for c in CANONICAL if c in df.columns]
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject physics-informed derived features."""
    out = df.copy()
    cols = set(out.columns)

    if {"width", "aspect_ratio"} <= cols:
        out["sidewall_height_mm"] = out["width"] * out["aspect_ratio"] / 100.0
    if {"load", "width"} <= cols:
        out["load_per_width"] = out["load"] / out["width"].replace(0, np.nan)
    if {"pressure", "load"} <= cols:
        out["pressure_load_ratio"] = out["pressure"] / out["load"].replace(0, np.nan)
    if "speed" in cols:
        out["speed_sq"] = out["speed"] ** 2
    if {"aspect_ratio", "pressure"} <= cols:
        out["aspect_pressure"] = out["aspect_ratio"] * out["pressure"]
    if {"width", "load", "pressure"} <= cols:
        out["contact_patch_area"] = out["width"] * (out["load"] / out["pressure"].replace(0, np.nan))
    if {"load", "width", "sidewall_height_mm"} <= set(out.columns):
        denom = out["width"] * out["sidewall_height_mm"]
        out["load_index"] = out["load"] / denom.replace(0, np.nan)
    if {"speed", "pressure"} <= cols:
        out["speed_pressure_ratio"] = out["speed"] / out["pressure"].replace(0, np.nan)
    if {"width", "rim_width"} <= cols:
        out["rim_width_ratio"] = out["width"] / (out["rim_width"] * 25.4).replace(0, np.nan)
    if {"load", "width", "aspect_ratio"} <= cols:
        denom = out["width"] * out["aspect_ratio"]
        out["normalized_load"] = out["load"] / denom.replace(0, np.nan)
    if {"load", "speed"} <= cols:
        out["load_speed_interact"] = out["load"] * out["speed"]
    if {"pressure", "speed_sq"} <= set(out.columns):
        out["pressure_speed_sq"] = out["pressure"] * out["speed_sq"]

    return out


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame, cols: list[str], z: float = 4.0) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out and pd.api.types.is_numeric_dtype(out[c]):
            mu, sd = out[c].mean(), out[c].std()
            if sd > 0:
                out = out[np.abs(out[c] - mu) <= z * sd]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    targets: tuple[str, ...] = TARGETS,
    as_float32: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Full pipeline → returns (X, Y, feat_cols, tgt_cols).

    Steps:
      1. Drop rows missing ALL targets.
      2. Outlier removal on targets (before imputation to avoid corrupt neighbours).
      3. Physics feature engineering.
      4. Adaptive KNN imputation (k ∝ dataset size).
      5. One-hot encode categoricals (dtype=int, fixes pandas 2.x bool bug).
      6. Optionally cast to float32 to halve memory.
    """
    present = [t for t in targets if t in df.columns]

    df = df.dropna(subset=present, how="all")
    df = remove_outliers(df, present, z=4.0)
    df = add_physics_features(df)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c in CAT_COLS]

    # Adaptive KNN: k scales with dataset size
    k = min(10, max(3, len(df) // 20))
    if num_cols:
        df[num_cols] = KNNImputer(n_neighbors=k).fit_transform(df[num_cols])

    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str)

    feat_df = pd.get_dummies(
        df.drop(columns=[t for t in targets if t in df.columns]),
        columns=cat_cols,
        drop_first=True,
        dtype=int,          # pandas 2.x returns bool by default; int is numeric
    )
    feat_df = feat_df.select_dtypes(include=[np.number])

    if as_float32:
        feat_df = feat_df.astype(np.float32)

    y_df = df[[t for t in targets if t in df.columns]]
    if as_float32:
        y_df = y_df.astype(np.float32)

    return feat_df, y_df, list(feat_df.columns), list(y_df.columns)


def scale(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    sc = StandardScaler()
    return sc.fit_transform(X), sc
