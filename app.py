# ==============================
# STREAMLIT APP - PRODUCTION READY
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
# SAMPLE DATA LOADER
# ------------------------------
def load_sample_data():
    return pd.read_csv("sample_test.csv")

use_sample = st.toggle("🧪 Use Sample Test File")

# ------------------------------
# FILE INPUT
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

df = None

if use_sample:
    try:
        df = load_sample_data()
        st.info("📂 Using sample_test.csv from repository")
    except Exception as e:
        st.error(f"Failed to load sample file: {e}")

elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully")
    except Exception as e:
        st.error(f"File read failed: {e}")

# ------------------------------
# MAIN PIPELINE
# ------------------------------
if df is not None:

    try:
        st.caption("Mode: Sample Data" if use_sample else "Mode: Uploaded File")

        # ------------------------------
        # LARGE FILE PROTECTION
        # ------------------------------
        if len(df) > 100000:
            st.warning("⚠️ Large file detected. Sampling 100k rows.")
            df = df.sample(100000, random_state=42)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # Debug (optional)
        st.write("Columns:", df.columns.tolist())

        # ------------------------------
        # VALIDATION
        # ------------------------------
        missing = set(FEATURES) - set(df.columns)

        if missing:
            st.error(f"❌ Missing columns: {list(missing)}")
            st.stop()

        X = df[FEATURES].copy()

        # ------------------------------
        # SMART NULL HANDLING
        # ------------------------------
        if X.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing values detected. Applying smart imputation.")

            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna("unknown")

        # ------------------------------
        # PREDICTION
        # ------------------------------
        preds = model.predict(X)
        df["Prediction"] = np.where(preds == 1, "Attack", "Benign")

        # Confidence scores
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            df["Confidence"] = probs

        st.subheader("🔍 Predictions")
        st.dataframe(df.head())

        # ------------------------------
        # SUMMARY
        # ------------------------------
        col1, col2 = st.columns(2)
        col1.metric("🟢 Benign", int((preds == 0).sum()))
        col2.metric("🔴 Attack", int((preds == 1).sum()))

        # Chart
        st.bar_chart(df["Prediction"].value_counts())

        # ------------------------------
        # DOWNLOAD
        # ------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "predictions.csv")

    except Exception as e:
        st.error(f"Main processing failed: {e}")
        st.stop()

    # ------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------
    st.subheader("🧠 Model Explainability")

    try:
        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42)

        # ------------------------------
        # HANDLE PIPELINE CORRECTLY
        # ------------------------------
        if hasattr(model, "named_steps"):

            model_for_shap = model.named_steps.get("model", model)

            if "preprocessor" in model.named_steps:
                X_sample_transformed = model.named_steps["preprocessor"].transform(X_sample)
            else:
                X_sample_transformed = X_sample

        else:
            model_for_shap = model
            X_sample_transformed = X_sample

        # ------------------------------
        # SHAP
        # ------------------------------
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_sample_transformed)

        if isinstance(shap_values, list):
            plot_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            plot_values = shap_values[:, :, 1]
        else:
            plot_values = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(plot_values, X_sample_transformed, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
