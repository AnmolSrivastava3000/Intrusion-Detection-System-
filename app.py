import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Load Model and Scaler
try:
    model = joblib.load('iot_ids_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

# EXACT 46 FEATURES
FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 
    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

st.set_page_config(page_title="IoT IDS Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Intrusion Detection System")
st.markdown("8th Semester Project: Real-time Network Traffic Analysis using Random Forest")

# --- SIDEBAR: QUICK TEST ---
st.sidebar.header("🚀 Quick Demo")
st.sidebar.info("Use the 70MB dataset already on GitHub for a fast test.")

if st.sidebar.button('Run Test on GitHub Dataset'):
    try:
        with st.spinner('Loading data from GitHub...'):
            # Loading the file you pushed with LFS
            df_git = pd.read_csv('part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv')
            
            # 1. Select Features & Convert to Numeric
            input_data = df_git[FEATURES].apply(pd.to_numeric, errors='coerce')
            
            # 2. Handle NaN and Inf
            input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            input_data.fillna(0, inplace=True)
            
            # 3. Predict
            if not input_data.empty:
                try:
                    X = input_data.values 
                    scaled_data = scaler.transform(X)
                    preds = model.predict(scaled_data)
                    
                    df_git['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in preds]
                    st.success(f"Successfully analyzed {len(df_git)} rows!")
                    
                    # Visualization
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Threats Detected", (preds == 1).sum())
                    with c2:
                        fig = px.pie(df_git, names='Status', color='Status', 
                                     color_discrete_map={'BENIGN':'#2ecc71','MALICIOUS':'#e74c3c'})
                        st.plotly_chart(fig)
                        
                    st.dataframe(df_git[['Status'] + FEATURES[:5]].head(100))
