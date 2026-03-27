# ==============================
# STREAMLIT APP -IMPROVED
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="IoT IDS", layout="wide")
st.title("🔐 IoT Intrusion Detection System")

# ------------------------------
# LOAD MODEL + FEATURES
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("ids_pipeline.pkl")

@st.cache_data
def load_features():
    with open("feature_config.json") as f:
        return json.load(f)

model = load_model()
config = load_features()
FEATURES = config["features"]

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully")

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        missing = set(FEATURES) - set(df.columns)

        if missing:
            st.error(f"❌ Missing columns: {list(missing)}")
            st.stop()

        X = df[FEATURES].copy()

        if X.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing values detected. Filling with 0.")
            X = X.fillna(0)

        preds = model.predict(X)
        df["Prediction"] = np.where(preds == 1, "Attack", "Benign")

        st.subheader("🔍 Predictions")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        col1.metric("🟢 Benign", int((preds == 0).sum()))
        col2.metric("🔴 Attack", int((preds == 1).sum()))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "predictions.csv")

    except Exception as e:
        st.error(f"Main processing failed: {e}")
        st.stop()

    # ------------------------------
    # SHAP (separate try block)
    # ------------------------------
    st.subheader("🧠 Model Explainability")

    try:
        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42)

        if hasattr(model, "named_steps"):
            model_for_shap = model.named_steps["model"]
        else:
            model_for_shap = model

        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            plot_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            plot_values = shap_values[:, :, 1]
        else:
            plot_values = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(plot_values, X_sample, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
