# 🛡️ Explainable IoT Intrusion Detection System (X-IDS)

abstract from paper -ABSTRACT
The increased use of Internet of Things (IoT) devices has
expanded attack surface. The resource-constrained nature of
these devices makes them highly susceptible to cyberthreats.
Generally, Signature based Intrusion detection system (IDS) is
used for surveillance on IoT devices. However, Signature-based
IDS are not well suited for novel, zero-day attacks. Anomalybased IDS have better record against zero-day attacks. This study
evaluates five machine learning models i.e.- Logistic Regression
(LR), Decision Tree (DT), Random Forest (RF), XGBoost and
Voting Ensemble. In this study, models are trained and evaluated
on subset (3 lakhs attack/benign type) of CICIoT2023 dataset.
This study implements a strategic 40:60 data balancing technique
(under sampling attack traffic) to ensure robust detection of both
benign and malicious patterns. This research aims to achieve a
balance between computational cost and performance with
machine learning for new age IoT security. Results indicate that
the Random Forest model achieved the highest accuracy of
99.66%, outperforming the other evaluated models.


While working on this project I added interpretable-SHAP (SHapley Additive exPlanations).As their was need to explain how this model not only pick single parameter but test and work on different parameter.Traditional Intrusion Detection Systems (IDS) often act as "black boxes," providing detection results without reasoning. This project addresses the "Black Box" problem by integrating Explainable AI (XAI). Our system classifies network traffic into Benign or Malicious and provides a breakdown of which network features (e.g., Packet Rate, Header Length) influenced the decision.

## 🚀 Live Demo
The application is deployed on Streamlit Cloud: https://bp2femsffh4yonhygceyny.streamlit.app/

## 📖 Project Overview

### Key Features:
* **Ensemble Modeling:** Uses a Voting Classifier (Random Forest + XGBoost + Logistic Regression).
* Random Forest scored highest while testing the models.
* **Real-time Explainability:** Integrated SHAP summary and bar plots for post-hoc interpretability.
* **Scalable Architecture:** Pre-processed using standard scaling and optimized for low-latency cloud inference.

## 🛠️ System Architecture
1. Data Ingestion: Loads IoT network traffic logs (CICIoT2023 sub part of Dataset).
2. Preprocessing: Feature selection (46 features) and Standardization via `StandardScaler`.
3. Inference Engine: Hybrid Ensemble model classifies traffic with ~99% accuracy.
4. XAI Layer: SHAP TreeExplainer generates local feature contribution charts.

## 📂 Repository Structure
* `app.py`: The Streamlit dashboard interface.
* `Intrusion-Detection-System.ipynb`: Training notebook including Data Balancing (40/60) and SHAP analysis.
* `iot_ids_model.pkl`: The serialized pre-trained Ensemble model.
* `scaler.pkl`: Serialized StandardScaler fitted on training data.
* `requirements.txt`: Necessary Python libraries for deployment.

## 🔧 Installation & Local Setup
1. Clone the repo:

Install dependencies:

Bash
pip install -r requirements.txt
Run the app:

Bash
streamlit run app.py

📊 Dataset Reference
This project utilizes the CICIoT2023 Dataset, a comprehensive benchmark for IoT security. We utilized a balanced subset of 300,000 samples to ensure model robustness across various attack vectors including DDoS, DoS, and Mirai botnets.

🎓 Author
Anmol Srivastava B.Tech Computer Science (2022-2026)
