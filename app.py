# ==============================
# STREAMLIT APP (IMPROVED)
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

        # ------------------------------
        # VALIDATION
        # ------------------------------
        missing = set(FEATURES) - set(df.columns)
        extra = set(df.columns) - set(FEATURES)

        if missing:
            st.error(f"❌ Missing columns: {list(missing)}")
            st.stop()

        # Keep only required columns in correct order
        X = df[FEATURES].copy()

        # Null handling
        if X.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing values detected. Filling with 0.")
            X = X.fillna(0)

        # ------------------------------
        # PREDICTION
        # ------------------------------
        preds = model.predict(X)

        df["Prediction"] = np.where(preds == 1, "Attack", "Benign")

        st.subheader("🔍 Predictions")
        st.dataframe(df.head())

        # ------------------------------
        # SUMMARY
        # ------------------------------
        col1, col2 = st.columns(2)

        attack_count = int((preds == 1).sum())
        benign_count = int((preds == 0).sum())

        col1.metric("🟢 Benign", benign_count)
        col2.metric("🔴 Attack", attack_count)

        # ------------------------------
        # DOWNLOAD RESULTS
        # ------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )
# ------------------------------
        # SHAP
        # ------------------------------
        st.subheader("🧠 Model Explainability")

        # Keep sample size small to ensure the web app stays fast
        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42)

        try:
            with st.spinner("Generating SHAP explanations..."):
                # 1. Extract the actual model if it's wrapped in a Pipeline
                if hasattr(model, "named_steps"):
                    model_for_shap = model.named_steps["model"]
                    # (Note: if you used a scaler in your pipeline, you would need to scale X_sample here first. 
                    # But since Random Forest doesn't need scaling, this is safe).
                else:
                    model_for_shap = model

                # 2. Use TreeExplainer for blazing fast performance
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values = explainer.shap_values(X_sample)

                # 3. Handle Binary Classification Output (Focus on Class 1: Attack)
                if isinstance(shap_values, list):
                    plot_values = shap_values[1] 
                elif len(shap_values.shape) == 3:
                    plot_values = shap_values[:, :, 1]
                else:
                    plot_values = shap_values

                # 4. Render plot safely in Streamlit
                plt.figure(figsize=(10, 6))
                shap.summary_plot(plot_values, X_sample, show=False)
                st.pyplot(plt.gcf())
                plt.clf() # Clear the figure to free up memory

        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
