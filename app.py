import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# 1. Load Model and Scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('iot_ids_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

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
st.markdown("8th Semester Project: Real-time Network Traffic Analysis with Explainable AI")

# --- SIDEBAR: QUICK TEST ---
st.sidebar.header("🚀 Quick Demo")
st.sidebar.info("Analyze the pre-loaded GitHub dataset.")

if st.sidebar.button('Run Test on GitHub Dataset'):
    try:
        with st.spinner('Loading data from GitHub...'):
            df_git = pd.read_csv('part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv', nrows=10000)
            
            # Clean and prepare
            input_data = df_git[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
            X_raw = input_data.values 
            
            # Scaling and Prediction
            X_scaled = scaler.transform(X_raw) 
            preds = model.predict(X_scaled)
            
            df_git['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in preds]
            st.success(f"Successfully analyzed {len(df_git)} rows!")
            
            # Dashboard Metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Threats Detected", int((preds == 1).sum()))
            with c2:
                fig = px.pie(df_git, names='Status', color='Status', 
                             color_discrete_map={'BENIGN':'#2ecc71','MALICIOUS':'#e74c3c'})
                st.plotly_chart(fig)
            
            st.dataframe(df_git[['Status'] + FEATURES[:5]].head(100))
            
            # Store in session for SHAP
            st.session_state['current_df'] = df_git
            st.session_state['current_X'] = X_scaled
            
    except Exception as e:
        st.sidebar.error(f"Execution Error: {e}")

st.divider()

# --- MAIN: MANUAL UPLOADER ---
st.subheader("📤 Upload Custom Traffic Logs")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, nrows=10000)
    
    try:
        input_data = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)

        if st.button('🚀 Analyze Uploaded File'):
            with st.spinner('Scanning for threats...'):
                X_raw_up = input_data.values
                X_scaled_up = scaler.transform(X_raw_up)
                predictions = model.predict(X_scaled_up)
                
                df['Status'] = ['MALICIOUS' if p == 1 else 'BENIGN' for p in predictions]
                
                st.success("Analysis Complete!")
                res1, res2 = st.columns(2)
                with res1:
                    st.metric("Total Packets", len(df))
                    st.metric("Threats Detected", int((predictions == 1).sum()))
                with res2:
                    fig2 = px.pie(df, names='Status', color='Status',
                                 color_discrete_map={'BENIGN':'#2ecc71','MALICIOUS':'#e74c3c'})
                    st.plotly_chart(fig2)

                st.dataframe(df[['Status'] + FEATURES[:5]])
                
                st.session_state['current_df'] = df
                st.session_state['current_X'] = X_scaled_up

    except Exception as e:
        st.error(f"Error: {e}")

# --- SHAP EXPLAINABILITY SECTION ---
if 'current_df' in st.session_state:
    st.divider()
    st.subheader("🔍 Explainable AI (SHAP) Interpretation")
    st.info("Select a packet from the analysis above to see WHY the model classified it as Malicious or Benign.")
    
    row_idx = st.number_input("Enter Row Index to explain:", min_value=0, max_value=len(st.session_state['current_df'])-1, value=0)
    
    if st.button("Explain Prediction"):
        with st.spinner("Generating SHAP Waterfall Plot..."):
            explainer = shap.TreeExplainer(model)
            # Explain the specific row
            specific_x = st.session_state['current_X'][row_idx].reshape(1, -1)
            shap_values = explainer.shap_values(specific_x)
            
            # Handle binary classification index
            # shap_values[1] is usually the 'Attack' class for Random Forest
            val_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(val_to_plot, specific_x, feature_names=FEATURES, plot_type="bar", show=False)
            st.pyplot(plt.gcf())
            plt.clf()
            
            st.write(f"**Insight:** This plot shows the top features contributing to the prediction for row {row_idx}. Large bars indicate features that heavily influenced the 'Malicious' classification.")
