import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('iot_ids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 
    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

st.set_page_config(page_title="IoT IDS", layout="wide")
st.title("🛡️ Explainable IoT Intrusion Detection")

# --- SIDEBAR: GITHUB DATASET TEST ---
st.sidebar.header("🚀 Quick Demo")
github_csv = 'part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'

if st.sidebar.button('Run Test on GitHub Dataset'):
    if os.path.exists(github_csv):
        df_git = pd.read_csv(github_csv, nrows=1000)
        input_data = df_git[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = scaler.transform(input_data.values)
        preds = model.predict(X_scaled)
        df_git['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in preds]
        
        st.session_state['result_df'] = df_git
        st.session_state['scaled_data'] = X_scaled
        st.sidebar.success("GitHub Data Loaded!")
    else:
        st.sidebar.error("CSV file not found on GitHub!")

# --- MAIN: DETECTION ---
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, nrows=1000)
    input_data = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_scaled = scaler.transform(input_data.values)
    
    if st.button("🚀 Run IDS Scan"):
        preds = model.predict(X_scaled)
        df['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in preds]
        st.session_state['result_df'] = df
        st.session_state['scaled_data'] = X_scaled

if 'result_df' in st.session_state:
    st.success("Scan Complete!")
    st.dataframe(st.session_state['result_df'][['Status'] + FEATURES[:5]])
    
    st.divider()
    st.header("🔍 Explainable AI Deep Dive")
    row_to_explain = st.number_input("Select Row Index:", 0, len(st.session_state['result_df'])-1, 0)
    
    if st.button("Explain Prediction"):
        with st.spinner("Calculating SHAP..."):
            explainer = shap.TreeExplainer(model)
            sample_scaled = st.session_state['scaled_data'][row_to_explain].reshape(1, -1)
            shap_v = explainer.shap_values(sample_scaled)
            
            # Binary logic
            row_shap = shap_v[1].flatten() if isinstance(shap_v, list) else shap_v.flatten()
            
            importance_df = pd.DataFrame({"Feature": FEATURES, "SHAP Value": row_shap})
            importance_df["abs_val"] = importance_df["SHAP Value"].abs()
            importance_df = importance_df.sort_values(by="abs_val", ascending=False).head(10)
            importance_df = importance_df.sort_values(by="SHAP Value")

            fig, ax = plt.subplots()
            colors = ["#ff0051" if x > 0 else "#008bfb" for x in importance_df["SHAP Value"]]
            ax.barh(importance_df["Feature"], importance_df["SHAP Value"], color=colors)
            st.pyplot(fig)
