import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Load Model and Scaler
model = joblib.load('iot_ids_model.pkl')
scaler = joblib.load('scaler.pkl')

# EXACT 46 FEATURES FROM YOUR PRINT OUTPUT
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
st.markdown("Upload network traffic logs to detect **Benign** vs **Malicious** activity.")

# File Uploader
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    try:
# 1. Filter for the exact features
        input_data = df[FEATURES]
        
        # 2. Convert all to numeric (in case some columns are read as objects)
        input_data = input_data.apply(pd.to_numeric, errors='coerce')

        # 3. Handle missing/infinite values (same as you did in v3.ipynb)
        input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_data.fillna(0, inplace=True) # Or use input_data.dropna() if you prefer
        # --------------------------------

        if st.button('🚀 Start Analysis'):
            with st.spinner('Scanning for threats...'):
                # Now the scaler should work without the TypeError
                scaled_input = scaler.transform(input_data)
                predictions = model.predict(scaled_input)
                
                # Mapping results
                df['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in predictions]
                
                # Visual Results
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

