"""
dashboard_app.py
----------------
Streamlit dashboard for the Tyre Traction Prediction & Simulation System.

Run:
    streamlit run dashboard_app.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from modules.data_processing import load_dataset, preprocess, add_physics_features
from modules.model_training import train_all, load_models, predict_row, parameter_trend_forecast
from modules.curve_model import gaussian_curve, pacejka_curve, estimate_pacejka_params, curve_insights
from modules.report_generator import build_pdf

ARTIFACTS = "artifacts"
DATA_PATH = os.environ.get("TYRE_DATA", "data/apollo.xlsx")
TARGETS = ["mu_peak", "slip_peak", "mu_lock"]

st.set_page_config(page_title="Tyre Traction Simulator", layout="wide",
                   page_icon="🛞")

# ---------- helpers ----------
@st.cache_resource(show_spinner=False)
def get_pipeline(force_retrain: bool = False):
    df = load_dataset(DATA_PATH)
    X, Y, feat_cols, tgt_cols = preprocess(df)
    needs = not all(os.path.exists(os.path.join(ARTIFACTS, f"xgb_{t}.joblib")) for t in tgt_cols)
    if needs or force_retrain:
        report = train_all(X, Y, out_dir=ARTIFACTS, n_iter=40)
    else:
        import json
        with open(os.path.join(ARTIFACTS, "report.json")) as f:
            report = __import__("json").load(f)
    models = load_models(tgt_cols, out_dir=ARTIFACTS)
    return df, X, Y, feat_cols, tgt_cols, models, report


def build_feature_row(template_cols, raw_inputs: dict) -> pd.DataFrame:
    """Align a single user-input row to the trained feature schema."""
    enriched = add_physics_features(pd.DataFrame([raw_inputs]))
    enriched = pd.get_dummies(enriched, dtype=int)  # int avoids bool/object cols
    row = pd.DataFrame(0.0, index=[0], columns=template_cols)
    for c in enriched.columns:
        if c in row.columns:
            row[c] = enriched[c].values[0]
    return row.astype(float)  # guarantee float64 — LightGBM rejects object dtype


# ---------- load ----------
with st.spinner("Loading dataset & models… (first run trains XGB + LightGBM + RF stacked ensemble — may take a minute)"):
    df, X, Y, feat_cols, tgt_cols, models, report = get_pipeline()

# ---------- sidebar ----------
st.sidebar.image("assets/apollo_logo.png", use_container_width=True)
st.sidebar.title("🛞 Tyre Inputs")
st.sidebar.caption("Manual values or auto-predict from dataset trends.")

auto = st.sidebar.toggle("Auto-predict from trends", value=False)

def trend_or(val_col: str, default: float) -> float:
    if auto and val_col in df.columns:
        return float(parameter_trend_forecast(df[val_col], 1)[0])
    return default

width = st.sidebar.number_input("Section width (mm)", 125, 355,
                                int(trend_or("width", 225)))
ar = st.sidebar.number_input("Aspect ratio", 25, 85, int(trend_or("aspect_ratio", 50)))
rim = st.sidebar.number_input("Rim diameter (inch)", 12, 24, int(trend_or("rim_dia", 18)))
rim_w = st.sidebar.number_input("Rim width (inch)", 4.0, 12.0,
                                float(trend_or("rim_width", 7.0)), step=0.5)
load = st.sidebar.number_input("Test load (N)", 1000, 12000,
                               int(trend_or("load", 6000)), step=100)
pressure = st.sidebar.number_input("Pressure (kPa)", 100, 400,
                                   int(trend_or("pressure", 220)), step=10)
speed = st.sidebar.number_input("Speed (kmph)", 20, 200,
                                int(trend_or("speed", 64)), step=5)

brands = sorted(df["brand"].dropna().unique().tolist()) if "brand" in df else []
brand = st.sidebar.selectbox("Brand", brands or ["unknown"], index=0)
test_types = sorted(df["test_type"].dropna().unique().tolist()) if "test_type" in df else []
test_type = st.sidebar.selectbox("Surface", test_types or ["Mu Slip - Dry"], index=0)

st.sidebar.divider()
compare = st.sidebar.checkbox("Compare with a second tyre")
if compare:
    st.sidebar.markdown("**Second tyre**")
    width2 = st.sidebar.number_input("Width #2", 125, 355, 205, key="w2")
    ar2 = st.sidebar.number_input("AR #2", 25, 85, 55, key="a2")
    pressure2 = st.sidebar.number_input("Pressure #2", 100, 400, 230, key="p2")
    load2 = st.sidebar.number_input("Load #2", 1000, 12000, 5500, key="l2")
    speed2 = st.sidebar.number_input("Speed #2", 20, 200, 64, key="s2")

# ---------- predict ----------
raw = dict(width=width, aspect_ratio=ar, rim_dia=rim, rim_width=rim_w,
           load=load, pressure=pressure, speed=speed,
           brand=brand, test_type=test_type)
row = build_feature_row(feat_cols, raw)
# First predict mu_peak, then pass it as cross-target feature for mu_lock.
preds = predict_row(models, row)
mu_peak = preds.get("mu_peak", 0.9)
slip_peak = max(min(preds.get("slip_peak", 0.12), 0.6), 0.02)
# Re-predict mu_lock with mu_peak as extra cross-target feature.
preds2 = predict_row(models, row, mu_peak_val=mu_peak)
mu_lock = preds2.get("mu_lock", mu_peak * 0.7)

s_g, mu_g = gaussian_curve(mu_peak, slip_peak)
s_p, mu_p, pace = pacejka_curve(mu_peak, slip_peak, mu_lock)

# ---------- layout ----------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("assets/apollo_logo.png", width=160)
with col_title:
    st.title("Tyre Traction Prediction & Simulation")
    st.caption("XGBoost predictions anchored into a Pacejka Magic-Formula curve.")

main, right = st.columns([3, 1.1])

with main:
    tab_curve, tab_anim, tab_compare, tab_importance, tab_data = st.tabs(
        ["µ–Slip Curve", "Animated build", "Compare", "Feature importance", "Dataset"]
    )

    with tab_curve:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s_g, y=mu_g, name="Gaussian", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=s_p, y=mu_p, name="Pacejka", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[slip_peak], y=[mu_peak], mode="markers+text",
                                 name="µpeak", text=["peak"], textposition="top center",
                                 marker=dict(size=12, color="crimson")))
        fig.update_layout(xaxis_title="Slip ratio", yaxis_title="µ",
                          template="plotly_white", height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tab_anim:
        placeholder = st.empty()
        if st.button("Animate Pacejka build"):
            for k in range(10, len(s_p) + 1, 6):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s_p[:k], y=mu_p[:k], line=dict(width=3, color="royalblue")))
                fig.update_layout(xaxis=dict(range=[0, s_p.max()]),
                                  yaxis=dict(range=[0, max(mu_p) * 1.1]),
                                  template="plotly_white", height=440,
                                  xaxis_title="Slip", yaxis_title="µ")
                placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.03)

    with tab_compare:
        if compare:
            raw2 = dict(width=width2, aspect_ratio=ar2, rim_dia=rim, rim_width=rim_w,
                        load=load2, pressure=pressure2, speed=speed2,
                        brand=brand, test_type=test_type)
            row2 = build_feature_row(feat_cols, raw2)
            p2 = predict_row(models, row2)
            mp2 = p2["mu_peak"]; sp2 = max(min(p2["slip_peak"], 0.6), 0.02)
            ml2 = p2.get("mu_lock", mp2 * 0.7)
            s2, m2, _ = pacejka_curve(mp2, sp2, ml2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_p, y=mu_p, name="Tyre #1"))
            fig.add_trace(go.Scatter(x=s2, y=m2, name="Tyre #2"))
            fig.update_layout(template="plotly_white", height=480,
                              xaxis_title="Slip", yaxis_title="µ")
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Tyre #1 µpeak", f"{mu_peak:.3f}", f"slip {slip_peak:.3f}")
            c2.metric("Tyre #2 µpeak", f"{mp2:.3f}", f"slip {sp2:.3f}")
        else:
            st.info("Enable 'Compare with a second tyre' in the sidebar.")

    with tab_importance:
        fi = report["targets"]["mu_peak"]["feature_importance"]
        fi_df = (pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
                 .sort_values("importance", ascending=True).tail(20))
        fig = go.Figure(go.Bar(x=fi_df["importance"], y=fi_df["feature"], orientation="h"))
        fig.update_layout(template="plotly_white", height=520,
                          title="Top features driving µpeak")
        st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.dataframe(df.head(200), use_container_width=True)

with right:
    st.subheader("Predictions")
    st.metric("µpeak", f"{mu_peak:.3f}")
    st.metric("Optimal slip", f"{slip_peak:.3f}")
    st.metric("µlock", f"{mu_lock:.3f}")
    st.divider()
    st.subheader("Pacejka coefficients")
    st.json(pace)
    st.divider()
    st.subheader("Model R² (hold-out)")
    lgbm_on = report.get("lgbm_available", False)
    st.caption(f"Ensemble: XGB + RF{'+ LightGBM' if lgbm_on else ''} → Ridge meta")
    for t in tgt_cols:
        m = report["targets"][t]
        r2_val = m['r2']
        badge = "🟢" if r2_val >= 0.85 else ("🟡" if r2_val >= 0.70 else "🔴")
        st.write(f"{badge} **{t}**  R²={r2_val:.3f}  RMSE={m['rmse']:.4f}")
    st.divider()
    st.subheader("Insight")
    insight = curve_insights(mu_peak, slip_peak, mu_lock)
    st.info(insight)

    if st.button("📄 Generate PDF report"):
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"tyre_report_{int(time.time())}.pdf")
        build_pdf(path, raw, {"µpeak": mu_peak, "slip_peak": slip_peak, "µlock": mu_lock,
                              **pace}, insight,
                  {"slip": s_p, "gauss": np.interp(s_p, s_g, mu_g), "pacejka": mu_p})
        with open(path, "rb") as f:
            st.download_button("Download report", f.read(), file_name=os.path.basename(path),
                               mime="application/pdf")

    if st.button("🔁 Retrain models"):
        get_pipeline.clear()
        get_pipeline(force_retrain=True)
        st.success("Retrained. Reload the page.")
