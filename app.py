import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Load Model and Scaler
model = joblib.load('iot_ids_model.pkl')
scaler = joblib.load('scaler.pkl')

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
st.markdown("Upload network traffic logs or use the **Quick Test** to detect **Benign** vs **Malicious** activity.")

# --- SIDEBAR QUICK TEST ---
st.sidebar.header("Quick Demo")
st.sidebar.info("Don't have a CSV? Run a test using the dataset already hosted on GitHub.")
if st.sidebar.button('🚀 Run Test on GitHub Dataset'):
    # Direct loading from the repo folder
    try:
        df_git = pd.read_csv('part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv')
        # Cleaning and Prediction logic
        input_data = df_git[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        scaled_data = scaler.transform(input_data)
        preds = model.predict(scaled_data)
        
        df_git['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in preds]
        
        st.success(f"Analysis Done! Processed {len(df_git)} rows from GitHub.")
        
        # Display results
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Threats Found", (preds == 1).sum())
        with c2:
            fig = px.pie(df_git, names='Status', color='Status', color_discrete_map={'BENIGN':'#2ecc71','MALICIOUS':'#e74c3c'})
            st.plotly_chart(fig)
    except FileNotFoundError:
        st.sidebar.error("File not found on server. Make sure the CSV name is correct in GitHub.")

st.divider()

# --- MAIN FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Your Own Network Traffic CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    try:
        input_data = df[FEATURES].apply(pd.to_numeric, errors='coerce')
        input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_data.fillna(0, inplace=True)

        if st.button('🚀 Start Analysis on Uploaded File'):
            with st.spinner('Scanning for threats...'):
                scaled_input = scaler.transform(input_data)
                predictions = model.predict(scaled_input)
                
                df['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in predictions]
                
                st.success("Analysis Complete!")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.metric("Total Rows", len(df))
                    threats = (predictions == 1).sum()
                    st.metric("Threats Detected", threats, delta=int(threats), delta_color="inverse")
                
                with c2:
                    fig = px.pie(df, names='Status', color='Status',
                                 color_discrete_map={'BENIGN':'#2ecc71','MALICIOUS':'#e74c3c'})
                    st.plotly_chart(fig)

                st.subheader("Analysis Log")
                st.dataframe(df[['Status'] + FEATURES[:5]])

    except KeyError as e:
        st.error(f"CSV Error: Missing columns {e}")
