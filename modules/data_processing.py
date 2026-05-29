"""
data_processing.py
------------------
Robust ingestion & preprocessing pipeline for tyre traction datasets.

Why this module exists:
- Real tyre test sheets have inconsistent column names, units in headers,
  newline characters, and missing values. We normalise everything to a
  canonical schema so downstream ML and physics code does not break.
- KNN imputation is preferred over mean/median because tyre parameters are
  strongly correlated (e.g. load <-> pressure, width <-> rim) so neighbour
  imputation preserves physical plausibility.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Canonical column names used everywhere downstream.
CANONICAL = {
    "width": ["sw", "section width", "width", "sw mm", "sw,mm"],
    "aspect_ratio": ["ar", "aspect ratio", "aspect"],
    "rim_dia": ["rim dia", "rim diameter", "rim dia inch", "rim"],
    "rim_width": ["rim width", "rim width inch"],
    "load": ["test load", "load", "fz", "vertical load"],
    "pressure": ["test ip", "ip", "pressure", "inflation pressure"],
    "speed": ["test speed", "speed", "vx"],
    "brand": ["brand", "manufacturer"],
    "pattern": ["pattern", "tread pattern"],
    "compound": ["tread cmpd", "compound", "tread compound"],
    "group": ["group"],
    "test_type": ["test details", "test type", "surface"],
    "trial": ["trial", "trial no"],
    "mu_peak": ["µpeak", "mu peak", "mupeak", "peak mu", "peak friction"],
    "slip_peak": ["sr @ µpeak", "slip peak", "sr peak", "slip at peak"],
    "mu_lock": ["µlock", "mu lock", "mulock", "lock mu"],
    "date": ["test date", "date"],
}


def _norm(s: str) -> str:
    """Normalise a column header for fuzzy matching."""
    s = str(s).lower().strip()
    s = s.replace("\n", " ").replace("μ", "µ")
    s = re.sub(r"[^a-z0-9µ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map raw columns to canonical names using normalised fuzzy matching.
    Prefers exact-normalised matches, then substring matches. Ensures every
    canonical name binds to a unique raw column.
    """
    raw = {c: _norm(c) for c in df.columns}
    used: set[str] = set()
    mapping: dict[str, str] = {}
    # Pass 1: exact normalised equality.
    for canon, aliases in CANONICAL.items():
        norm_aliases = [_norm(a) for a in aliases]
        for raw_col, norm in raw.items():
            if raw_col in used:
                continue
            if norm in norm_aliases:
                mapping[canon] = raw_col
                used.add(raw_col)
                break
    # Pass 2: substring containment for anything still unmapped.
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


def load_dataset(path: str, sheet: str | int = 0) -> pd.DataFrame:
    """Load the Excel sheet and rename columns to canonical schema."""
    df = pd.read_excel(path, sheet_name=sheet)
    mapping = detect_columns(df)
    df = df.rename(columns={v: k for k, v in mapping.items()})
    keep = [c for c in CANONICAL if c in df.columns]
    return df[keep].copy()


def remove_outliers(df: pd.DataFrame, cols: list[str], z: float = 4.0) -> pd.DataFrame:
    """Drop rows whose numeric columns lie beyond z standard deviations.
    Conservative z=4 so we only kill clear sensor glitches, not real variance.
    """
    out = df.copy()
    for c in cols:
        if c in out and pd.api.types.is_numeric_dtype(out[c]):
            mu, sd = out[c].mean(), out[c].std()
            if sd and sd > 0:
                out = out[np.abs(out[c] - mu) <= z * sd]
    return out.reset_index(drop=True)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject physics-informed derived features.

    Original features:
      sidewall_height_mm = width * AR/100  (geometric tyre sidewall)
      load_per_width     = Fz / SW         (contact-patch loading proxy)
      pressure_load_ratio= IP / Fz         (stiffness vs load - affects µ)
      speed_sq           = V^2             (kinetic / aero / hysteresis term)
      aspect_pressure    = AR * IP         (carcass stiffness proxy)

    Additional enhanced features:
      contact_patch_area   ≈ width * (load / pressure)  (normal load / contact pressure)
      load_index           = load / (width * sidewall_height_mm)  (volumetric loading)
      speed_pressure_ratio = speed / pressure  (aero-thermal proxy)
      rim_width_ratio      = width / (rim_width * 25.4)  (tyre stretch / profile index)
      normalized_load      = load / (width * aspect_ratio)  (contact pressure proxy)
      load_speed_interact  = load * speed  (cross-term: load x speed interaction)
      pressure_speed_sq    = pressure * speed^2  (compound kinetic term)
    """
    out = df.copy()

    # --- original features ---
    if {"width", "aspect_ratio"}.issubset(out.columns):
        out["sidewall_height_mm"] = out["width"] * out["aspect_ratio"] / 100.0
    if {"load", "width"}.issubset(out.columns):
        out["load_per_width"] = out["load"] / out["width"].replace(0, np.nan)
    if {"pressure", "load"}.issubset(out.columns):
        out["pressure_load_ratio"] = out["pressure"] / out["load"].replace(0, np.nan)
    if "speed" in out.columns:
        out["speed_sq"] = out["speed"] ** 2
    if {"aspect_ratio", "pressure"}.issubset(out.columns):
        out["aspect_pressure"] = out["aspect_ratio"] * out["pressure"]

    # --- enhanced features ---
    if {"width", "load", "pressure"}.issubset(out.columns):
        out["contact_patch_area"] = out["width"] * (
            out["load"] / out["pressure"].replace(0, np.nan)
        )
    if {"load", "width", "sidewall_height_mm"}.issubset(out.columns):
        denom = out["width"] * out["sidewall_height_mm"]
        out["load_index"] = out["load"] / denom.replace(0, np.nan)
    if {"speed", "pressure"}.issubset(out.columns):
        out["speed_pressure_ratio"] = out["speed"] / out["pressure"].replace(0, np.nan)
    if {"width", "rim_width"}.issubset(out.columns):
        rim_mm = out["rim_width"] * 25.4
        out["rim_width_ratio"] = out["width"] / rim_mm.replace(0, np.nan)
    if {"load", "width", "aspect_ratio"}.issubset(out.columns):
        denom = out["width"] * out["aspect_ratio"]
        out["normalized_load"] = out["load"] / denom.replace(0, np.nan)
    if {"load", "speed"}.issubset(out.columns):
        out["load_speed_interact"] = out["load"] * out["speed"]
    if {"pressure", "speed_sq"}.issubset(out.columns):
        out["pressure_speed_sq"] = out["pressure"] * out["speed_sq"]

    return out


def preprocess(
    df: pd.DataFrame,
    targets: tuple[str, ...] = ("mu_peak", "slip_peak", "mu_lock"),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Full preprocess pipeline: impute, engineer, encode, return X,y matrices.

    Order:
    1. Drop rows missing ALL targets (truly unlabelled).
    2. Remove target outliers BEFORE imputation so corrupt rows don't pollute
       neighbour lookups.
    3. Physics feature engineering.
    4. Adaptive KNN imputation (k scales with dataset size).
    5. One-hot encode categoricals.
    """
    present_targets = [t for t in targets if t in df.columns]

    # Step 1: drop rows that have no target at all
    df = df.dropna(subset=present_targets, how="all")

    # Step 2: outlier removal on target columns BEFORE imputation
    df = remove_outliers(df, present_targets, z=4.0)

    # Step 3: physics features
    df = add_physics_features(df)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns
                if c in ("brand", "pattern", "compound", "group", "test_type")]

    # Step 4: adaptive KNN imputation
    n_rows = len(df)
    k = min(10, max(3, n_rows // 20))
    if num_cols:
        imp = KNNImputer(n_neighbors=k)
        df[num_cols] = imp.fit_transform(df[num_cols])

    # Fill categorical NaNs with explicit "unknown" so OHE can encode them.
    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str)

    feat_df = pd.get_dummies(
        df.drop(columns=[t for t in targets if t in df.columns]),
        columns=cat_cols,
        drop_first=True,
        dtype=int,          # pandas 2.x returns bool by default; int passes select_dtypes
    )
    feat_df = feat_df.select_dtypes(include=[np.number])
    y_df = df[[t for t in targets if t in df.columns]]
    return feat_df, y_df, list(feat_df.columns), list(y_df.columns)


def scale(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    sc = StandardScaler()
    return sc.fit_transform(X), sc
