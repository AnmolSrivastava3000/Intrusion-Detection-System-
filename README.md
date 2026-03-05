# Intrusion-Detection-System-
IDS ,anomaly based ,logistic regression ,CICIOT2023 ,for IOT devices (resource constrained)


# 🛡️ IoT Intrusion Detection System (IDS)
**8th Semester Major Project | B.Tech CSE**

This project is a real-time Network Intrusion Detection System designed specifically for IoT environments. It uses Machine Learning to classify network traffic as **Benign** (Safe) or **Malicious** (Attack) with high precision.

## 🚀 Live Demo
Check out the live web application here: `[PASTE_YOUR_STREAMLIT_LINK_HERE]`

## 📊 Project Overview
- **Dataset:** CICIoT2023 (Comprehensive IoT Cybersecurity Dataset).
- **Model:** Random Forest Classifier (Optimized via GridSearchCV).
- **Accuracy:** ~99.6% on testing data.
- **Features:** Analysis of 46 distinct network traffic features (Flow duration, Header Length, Protocol Types, etc.).

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Machine Learning:** Scikit-learn, Joblib
- **Data Handling:** Pandas, Numpy
- **Web Framework:** Streamlit (Cloud Deployment)
- **Visualizations:** Plotly Express

## 📂 Repository Structure
- `app.py`: The main Streamlit application script.
- `iot_ids_model.pkl`: The trained Random Forest model.
- `scaler.pkl`: The StandardScaler object for data normalization.
- `requirements.txt`: List of dependencies for cloud deployment.
- `*.csv`: Sample test data stored via **Git LFS**.

## 📝 Features Analyzed
The model inspects 46 features from network packets, including:
- **Flag Numbers:** FIN, SYN, RST, PSH, ACK, ECE, CWR.
- **Protocols:** TCP, UDP, HTTP, HTTPS, DNS, ICMP, etc.
- **Statistical Metrics:** Min, Max, AVG, Std, Variance.

## 👨‍💻 Author
**Anmol Srivastava** B.Tech CSE Student @ Galgotias University

---
*Developed as part of the 8th Semester Research Project on Intrusion Detection Systems.*
