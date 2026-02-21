import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Model and Scaler
model = joblib.load('iot_ids_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🛡️ IoT Intrusion Detection System")
st.write("Real-time ML analysis for IoT network traffic.")

# 2. Sidebar for Manual Input (using the top features from research)
st.sidebar.header("Input Network Features")
def user_input_features():
# Replace these with the specific feature names from CICIoT2023 subset

    rate = st.sidebar.number_input('Rate', value=0.0)
    protocol_type = st.sidebar.number_input('Protocol Type', value=1.0)
    header_length = st.sidebar.number_input('Header Length', value=0.0)
    
    data = {'Rate': rate,
            'ProtocolType': protocol_type,
            'Header_Length': header_length}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. Prediction Logic
if st.button('Analyze Traffic'):
    # Standardize the input exactly like you did in training
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 0: # Adjust based on your LabelEncoder
        st.success("✅ Benign Traffic Detected")
    else:
        st.error("⚠️ Malicious Attack Detected!")