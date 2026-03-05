import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Load Model and Scaler
# Ensure these files are in your GitHub repository
model = joblib.load('iot_ids_model.pkl')
scaler = joblib.load('scaler.pkl')

# Complete list of 47 features from your v3.ipynb (CICIoT2023)
FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 
    'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight', 'Magnitue_Avg'
]

# Set Page Config for a professional look
st.set_page_config(page_title="IoT IDS Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Intrusion Detection System")
st.markdown("""
This system uses Machine Learning to detect malicious traffic in IoT networks.
Upload a network traffic log (CSV) to analyze potential threats.
""")

# Sidebar info
st.sidebar.header("Model Information")
st.sidebar.info("Model: Random Forest/XGBoost Ensemble\nDataset: CICIoT2023\nFeatures: 47 Network Parameters")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # 3. Pre-processing and Feature Alignment
    try:
        # We only take the 47 features the model expects
        input_data = df[FEATURES]
        
        if st.button('🚀 Start Security Analysis'):
            with st.spinner('Analyzing traffic patterns...'):
                # Scale the data
                scaled_input = scaler.transform(input_data)
                
                # Make Predictions
                predictions = model.predict(scaled_input)
                
                # Add predictions to the display dataframe
                df['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in predictions]
                
                # 4. Results & Visualization
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Packets Analyzed", len(df))
                    malicious_count = (predictions == 1).sum()
                    st.metric("Threats Detected", malicious_count, delta=int(malicious_count), delta_color="inverse")
                
                with col2:
                    # Professional Chart
                    fig = px.pie(df, names='Status', title='Traffic Composition', 
                                 color='Status', color_discrete_map={'BENIGN':'green', 'MALICIOUS':'red'})
                    st.plotly_chart(fig)

                st.subheader("Detailed Logs")
                st.dataframe(df[['Status'] + FEATURES[:5]])
                
                # Option to download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Analysis Report", data=csv, file_name="ids_report.csv", mime="text/csv")

    except KeyError as e:
        st.error(f"Error: Your CSV is missing required columns. Please ensure it contains: {e}")
else:
    st.warning("Waiting for CSV file upload...")
