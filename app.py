import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="IoT IDS", layout="wide")
st.title("🛡️ Explainable IoT Intrusion Detection")

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("iot_ids_model.pkl")
    scaler = joblib.load("scaler.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

model, scaler, explainer = load_assets()

# -----------------------------
# FEATURE LIST
# -----------------------------
FEATURES = [
    'flow_duration','Header_Length','Protocol Type','Duration','Rate','Srate','Drate',
    'fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number','ack_flag_number',
    'ece_flag_number','cwr_flag_number','ack_count','syn_count','fin_count','urg_count',
    'rst_count','HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC','TCP','UDP','DHCP',
    'ARP','ICMP','IPv','LLC','Tot sum','Min','Max','AVG','Std','Tot size','IAT',
    'Number','Magnitue','Radius','Covariance','Variance','Weight'
]

# -----------------------------
# SIDEBAR DEMO DATA
# -----------------------------
st.sidebar.header("🚀 Quick Demo")

github_csv = "part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"

if st.sidebar.button("Run Test on GitHub Dataset"):

    if os.path.exists(github_csv):

        df_git = pd.read_csv(github_csv, nrows=1000)

        input_data = df_git[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

        X_scaled = scaler.transform(input_data.values)

        preds = model.predict(X_scaled)

        df_git["Status"] = ["MALICIOUS" if p == 1 else "BENIGN" for p in preds]

        st.session_state["result_df"] = df_git
        st.session_state["scaled_data"] = X_scaled

        st.sidebar.success("Dataset Loaded!")

    else:
        st.sidebar.error("CSV not found")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file, nrows=1000)

    input_data = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_scaled = scaler.transform(input_data.values)

    if st.button("🚀 Run IDS Scan"):

        preds = model.predict(X_scaled)

        df["Status"] = ["MALICIOUS" if p == 1 else "BENIGN" for p in preds]

        st.session_state["result_df"] = df
        st.session_state["scaled_data"] = X_scaled


# -----------------------------
# RESULTS
# -----------------------------
if "result_df" in st.session_state:

    st.success("Scan Complete!")

    result_df = st.session_state["result_df"]

    st.dataframe(result_df[["Status"] + FEATURES[:5]])

    # -----------------------------
    # PREDICTION DISTRIBUTION
    # -----------------------------
    st.subheader("📊 Prediction Distribution")

    counts = result_df["Status"].value_counts()

    st.bar_chart(counts)

    st.divider()

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.header("🔍 Explainable AI Deep Dive")

    row_to_explain = st.number_input(
        "Select Row Index",
        min_value=0,
        max_value=len(result_df) - 1,
        value=0
    )

    if st.button("Explain Prediction"):

        with st.spinner("Calculating SHAP explanation..."):

            sample_scaled = st.session_state["scaled_data"][row_to_explain].reshape(1, -1)

            shap_values = explainer.shap_values(sample_scaled)

            # Handle SHAP output
            if isinstance(shap_values, list):
                row_shap = shap_values[1].flatten()
            else:
                row_shap = shap_values.flatten()

            # Fix double-length issue (92 instead of 46)
            if len(row_shap) == len(FEATURES) * 2:
                row_shap = row_shap.reshape(2, -1)[1]

            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP Value": row_shap
            })

            importance_df["abs_val"] = importance_df["SHAP Value"].abs()

            importance_df = importance_df.sort_values(
                by="abs_val",
                ascending=False
            ).head(10)

            importance_df = importance_df.sort_values(by="SHAP Value")

            # -----------------------------
            # PLOT
            # -----------------------------
            fig, ax = plt.subplots(figsize=(8,4))

            colors = [
                "#ff0051" if x > 0 else "#008bfb"
                for x in importance_df["SHAP Value"]
            ]

            ax.barh(
                importance_df["Feature"],
                importance_df["SHAP Value"],
                color=colors
            )

            ax.set_xlabel("Impact on Attack Prediction")
            ax.set_title("Local SHAP Explanation")

            st.pyplot(fig)
            st.markdown(f"""
**Interpretation:** * Features in **<span style='color:#ff0051'>Red</span>** increased the probability of an Attack.
* Features in **<span style='color:#008bfb'>Blue</span>** decreased the probability (pushed toward Benign).
* Current Prediction for Row {row_to_explain}: **{result_df.iloc[row_to_explain]['Status']}**
""", unsafe_allow_html=True)
