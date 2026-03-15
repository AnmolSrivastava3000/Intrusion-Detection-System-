import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('iot_ids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Use the feature names exactly as they appeared in your notebook
FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 
    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

st.title("🛡️ Explainable IoT Intrusion Detection")

# --- STEP 1: DETECTION ---
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, nrows=1000)
    
    # Preprocess
    input_data = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_scaled = scaler.transform(input_data.values)
    
    if st.button("🚀 Run IDS Scan"):
        preds = model.predict(X_scaled)
        df['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in preds]
        
        # Save to session state for the SHAP section
        st.session_state['result_df'] = df
        st.session_state['scaled_data'] = X_scaled
        
        st.success("Scan Complete!")
        st.dataframe(df[['Status'] + FEATURES[:5]])

# --- STEP 2: EXPLAINABILITY (The Research Part) ---
if 'result_df' in st.session_state:
    st.divider()
    st.header("🔍 Explainable AI Deep Dive")
    st.info("Select a row from the scan results above to see the SHAP explanation.")
    
    row_to_explain = st.number_input("Select Row Index:", 0, len(st.session_state['result_df'])-1, 0)
    
    if st.button("Explain Prediction"):
        with st.spinner("Calculating SHAP values..."):
            # Setup Explainer
            explainer = shap.TreeExplainer(model)
            
            # Get data for that row
            sample_scaled = st.session_state['scaled_data'][row_to_explain].reshape(1, -1)
            
            # Calculate SHAP
            shap_v = explainer.shap_values(sample_scaled)
            
            # Handle binary output based on your working notebook logic
            if isinstance(shap_v, list):
                row_shap = shap_v[1].flatten()
                base_val = explainer.expected_value[1]
            else:
                row_shap = shap_v.flatten()
                base_val = explainer.expected_value

            # Build the same Importance DF from your notebook
            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP Value": row_shap
            })
            importance_df["abs_val"] = importance_df["SHAP Value"].abs()
            importance_df = importance_df.sort_values(by="abs_val", ascending=False).head(10)
            importance_df = importance_df.sort_values(by="SHAP Value")

            # Plot
            fig, ax = plt.subplots()
            colors = ["#ff0051" if x > 0 else "#008bfb" for x in importance_df["SHAP Value"]]
            ax.barh(importance_df["Feature"], importance_df["SHAP Value"], color=colors)
            ax.set_title(f"Why Row {row_to_explain} was flagged")
            
            st.pyplot(fig)
            st.write(f"**Prediction Interpretation:** Red bars show features that pushed the model toward an 'Attack' decision, while blue bars pushed it toward 'Benign'.")
