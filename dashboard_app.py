"""
dashboard_app.py
----------------
Tyre Traction Prediction & Simulation System — Production Dashboard.

Startup contract:
  ✓ NEVER trains models.
  ✓ NEVER runs CV or hyperparameter search.
  ✓ Loads pre-trained artifacts from artifacts/ (run train_models.py offline).
  ✓ Target cold-start: < 5 seconds on Render free tier.

If artifacts are missing → friendly error with instructions.

Run:
    streamlit run dashboard_app.py
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from modules.data_processing import add_physics_features
from modules.model_training  import (
    load_models, load_feature_cols, load_report,
    artifacts_exist, predict_all, parameter_trend_forecast,
)
from modules.curve_model      import gaussian_curve, pacejka_curve, curve_insights
from modules.report_generator import build_pdf

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS   = "artifacts"
TARGETS     = ["mu_peak", "slip_peak", "mu_lock"]
DATA_CSV    = "data/apollo.csv"
DATA_XLSX   = "data/apollo.xlsx"

st.set_page_config(
    page_title="Tyre Traction Simulator",
    layout="wide",
    page_icon="🛞",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders  (run exactly once per server restart)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_models():
    """Load all 12 joblib artifacts (4 models × 3 targets). Runs once."""
    return load_models(TARGETS, ARTIFACTS)


@st.cache_resource(show_spinner=False)
def _load_feat_cols():
    return load_feature_cols(ARTIFACTS)


@st.cache_resource(show_spinner=False)
def _load_report():
    return load_report(ARTIFACTS)


@st.cache_data(show_spinner=False, ttl=3_600)
def _load_dataset() -> pd.DataFrame:
    """Load dataset. Prefers CSV (faster). Falls back to XLSX."""
    if os.path.exists(DATA_CSV):
        return pd.read_csv(DATA_CSV)
    if os.path.exists(DATA_XLSX):
        from modules.data_processing import load_dataset
        return load_dataset(DATA_XLSX)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Artifact guard  (user-friendly error if train_models.py wasn't run)
# ─────────────────────────────────────────────────────────────────────────────
ok, missing_files = artifacts_exist(TARGETS, ARTIFACTS)
if not ok:
    st.error("### ⚠️ Model artifacts not found")
    st.markdown(
        "The dashboard requires pre-trained model files. "
        "Run the training script **once** on your local machine:\n\n"
        "```bash\npython train_models.py\n```\n\n"
        "Then commit the `artifacts/` folder and redeploy.\n\n"
        "**Missing files:**"
    )
    for f in missing_files:
        st.code(f)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Load everything (all cached → near-instant after first load)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    all_models = _load_models()
    feat_cols  = _load_feat_cols()
    report     = _load_report()
    df         = _load_dataset()

tgt_cols = [t for t in TARGETS if t in report.get("targets", {})]


# ─────────────────────────────────────────────────────────────────────────────
# Feature-row builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(raw: dict) -> np.ndarray:
    """
    Build a (1, n_features) float32 numpy array aligned to feat_cols.

    Steps:
      1. Add physics-derived features.
      2. One-hot encode brand/test_type.
      3. Align to training feature schema (zero-fill unseen columns).
      4. Cast to float32.
    """
    enriched = add_physics_features(pd.DataFrame([raw]))
    enriched = pd.get_dummies(enriched, dtype=int)
    row = pd.DataFrame(0.0, index=[0], columns=feat_cols)
    for c in enriched.columns:
        if c in row.columns:
            row[c] = enriched[c].values[0]
    return row.astype(np.float32).values   # (1, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists("assets/apollo_logo.png"):
    st.sidebar.image("assets/apollo_logo.png", use_container_width=True)

st.sidebar.title("🛞 Tyre Inputs")
st.sidebar.caption("Adjust parameters or use auto-predict from dataset trends.")

auto = st.sidebar.toggle("Auto-predict from dataset trends", value=False)

def trend_val(col: str, default: float) -> float:
    if auto and not df.empty and col in df.columns:
        return float(parameter_trend_forecast(df[col], 1)[0])
    return default

width    = st.sidebar.number_input("Section width (mm)",    125, 355,
                                   int(trend_val("width", 225)))
ar       = st.sidebar.number_input("Aspect ratio",           25,  85,
                                   int(trend_val("aspect_ratio", 50)))
rim      = st.sidebar.number_input("Rim diameter (inch)",    12,  24,
                                   int(trend_val("rim_dia", 18)))
rim_w    = st.sidebar.number_input("Rim width (inch)",      4.0, 12.0,
                                   float(trend_val("rim_width", 7.0)), step=0.5)
load     = st.sidebar.number_input("Test load (N)",        1000, 12000,
                                   int(trend_val("load", 6000)), step=100)
pressure = st.sidebar.number_input("Pressure (kPa)",        100,  400,
                                   int(trend_val("pressure", 220)), step=10)
speed    = st.sidebar.number_input("Speed (km/h)",           20,  200,
                                   int(trend_val("speed", 64)), step=5)

brands     = sorted(df["brand"].dropna().unique().tolist()) if (not df.empty and "brand" in df) else ["unknown"]
test_types = sorted(df["test_type"].dropna().unique().tolist()) if (not df.empty and "test_type" in df) else ["Mu Slip - Dry"]
brand     = st.sidebar.selectbox("Brand",   brands,     index=0)
test_type = st.sidebar.selectbox("Surface", test_types, index=0)

st.sidebar.divider()
compare = st.sidebar.checkbox("Compare with a second tyre")
if compare:
    st.sidebar.markdown("**Second tyre**")
    width2    = st.sidebar.number_input("Width #2",    125, 355, 205, key="w2")
    ar2       = st.sidebar.number_input("AR #2",        25,  85,  55, key="a2")
    pressure2 = st.sidebar.number_input("Pressure #2", 100, 400, 230, key="p2")
    load2     = st.sidebar.number_input("Load #2",    1000, 12000, 5500, key="l2")
    speed2    = st.sidebar.number_input("Speed #2",    20,  200,   64,  key="s2")

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
raw = dict(width=width, aspect_ratio=ar, rim_dia=rim, rim_width=rim_w,
           load=load, pressure=pressure, speed=speed,
           brand=brand, test_type=test_type)
X_row = build_feature_row(raw)
preds = predict_all(all_models, X_row, feat_cols)

mu_peak   = preds["mu_peak"]
slip_peak = preds["slip_peak"]
mu_lock   = preds["mu_lock"]

s_g, mu_g          = gaussian_curve(mu_peak, slip_peak)
s_p, mu_p, pace    = pacejka_curve(mu_peak, slip_peak, mu_lock)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("assets/apollo_logo.png"):
        st.image("assets/apollo_logo.png", width=140)
with col_title:
    st.title("Tyre Traction Prediction & Simulation")
    st.caption(
        "OOF-stacked ensemble (XGB + RF + LightGBM → Ridge) | "
        "Predictions anchored to Pacejka Magic Formula"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
main_col, right_col = st.columns([3, 1.1])

with main_col:
    tab_curve, tab_anim, tab_compare, tab_fi, tab_data = st.tabs([
        "µ–Slip Curve", "Animated build", "Compare", "Feature importance", "Dataset",
    ])

    # ── µ–Slip curve ────────────────────────────────────────────────────────
    with tab_curve:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s_g, y=mu_g, name="Gaussian baseline",
            line=dict(dash="dash", color="#94a3b8"),
        ))
        fig.add_trace(go.Scatter(
            x=s_p, y=mu_p, name="Pacejka MF",
            line=dict(width=3, color="#3b82f6"),
        ))
        fig.add_trace(go.Scatter(
            x=[slip_peak], y=[mu_peak],
            mode="markers+text",
            name="µpeak",
            text=["peak"],
            textposition="top center",
            marker=dict(size=12, color="#ef4444"),
        ))
        fig.add_trace(go.Scatter(
            x=[1.0], y=[mu_lock],
            mode="markers+text",
            name="µlock",
            text=["lock"],
            textposition="top left",
            marker=dict(size=10, color="#f97316", symbol="diamond"),
        ))
        fig.update_layout(
            xaxis_title="Slip ratio (–)",
            yaxis_title="Friction coefficient µ",
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Animated build ──────────────────────────────────────────────────────
    with tab_anim:
        anim_ph = st.empty()
        if st.button("▶ Animate Pacejka build"):
            for k in range(10, len(s_p) + 1, 6):
                af = go.Figure()
                af.add_trace(go.Scatter(
                    x=s_p[:k], y=mu_p[:k],
                    line=dict(width=3, color="#3b82f6"),
                ))
                af.update_layout(
                    xaxis=dict(range=[0, float(s_p.max())]),
                    yaxis=dict(range=[0, float(max(mu_p)) * 1.15]),
                    template="plotly_white",
                    height=440,
                    xaxis_title="Slip", yaxis_title="µ",
                )
                anim_ph.plotly_chart(af, use_container_width=True)
                time.sleep(0.03)

    # ── Compare ─────────────────────────────────────────────────────────────
    with tab_compare:
        if compare:
            raw2 = dict(width=width2, aspect_ratio=ar2, rim_dia=rim,
                        rim_width=rim_w, load=load2, pressure=pressure2,
                        speed=speed2, brand=brand, test_type=test_type)
            X_row2  = build_feature_row(raw2)
            preds2  = predict_all(all_models, X_row2, feat_cols)
            mp2     = preds2["mu_peak"]
            sp2     = preds2["slip_peak"]
            ml2     = preds2["mu_lock"]
            s2, m2, _ = pacejka_curve(mp2, sp2, ml2)

            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Scatter(x=s_p, y=mu_p, name="Tyre #1",
                                         line=dict(width=3, color="#3b82f6")))
            cmp_fig.add_trace(go.Scatter(x=s2, y=m2, name="Tyre #2",
                                         line=dict(width=3, color="#f97316")))
            cmp_fig.update_layout(template="plotly_white", height=480,
                                  xaxis_title="Slip", yaxis_title="µ")
            st.plotly_chart(cmp_fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Tyre #1 µpeak", f"{mu_peak:.3f}", f"slip {slip_peak:.3f}")
            c2.metric("Tyre #2 µpeak", f"{mp2:.3f}", f"slip {sp2:.3f}")
            c1.metric("Tyre #1 µlock", f"{mu_lock:.3f}")
            c2.metric("Tyre #2 µlock", f"{ml2:.3f}")
        else:
            st.info("Enable **Compare with a second tyre** in the sidebar.")

    # ── Feature importance ───────────────────────────────────────────────────
    with tab_fi:
        fi = report["targets"].get("mu_peak", {}).get("feature_importance", {})
        if fi:
            fi_df = (
                pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
                .sort_values("importance", ascending=True)
                .tail(20)
            )
            fi_fig = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                marker_color="#3b82f6",
            ))
            fi_fig.update_layout(template="plotly_white", height=520,
                                 title="Top 20 features driving µpeak (XGB importance)")
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Feature importance not available.")

    # ── Dataset preview ──────────────────────────────────────────────────────
    with tab_data:
        if df.empty:
            st.warning("Dataset file not found.")
        else:
            st.dataframe(df.head(300), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Right panel
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    st.subheader("Predictions")
    st.metric("µ peak friction",  f"{mu_peak:.3f}")
    st.metric("Optimal slip ratio", f"{slip_peak:.3f}")
    st.metric("µ lock (ABS ref)", f"{mu_lock:.3f}")

    st.divider()
    st.subheader("Pacejka coefficients")
    st.json({k: round(v, 4) for k, v in pace.items()})

    st.divider()
    st.subheader("Model metrics (hold-out)")
    lgbm_on = report.get("lgbm_available", True)
    st.caption(
        f"OOF Stack: XGB + RF{' + LGB' if lgbm_on else ''} → Ridge"
    )
    for t in tgt_cols:
        m      = report["targets"].get(t, {})
        r2_val = m.get("r2", 0.0)
        badge  = "🟢" if r2_val >= 0.85 else ("🟡" if r2_val >= 0.70 else "🔴")
        st.write(f"{badge} **{t}**  R²={r2_val:.3f}  RMSE={m.get('rmse', 0):.4f}")

    st.divider()
    st.subheader("Engineering insight")
    st.info(curve_insights(mu_peak, slip_peak, mu_lock))

    # PDF report
    if st.button("📄 Generate PDF report"):
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"tyre_report_{int(time.time())}.pdf")
        try:
            build_pdf(
                path, raw,
                {"µpeak": mu_peak, "slip_peak": slip_peak, "µlock": mu_lock, **pace},
                curve_insights(mu_peak, slip_peak, mu_lock),
                {"slip": s_p, "gauss": np.interp(s_p, s_g, mu_g), "pacejka": mu_p},
            )
            with open(path, "rb") as fh:
                st.download_button(
                    "⬇ Download PDF", fh.read(),
                    file_name=os.path.basename(path),
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
