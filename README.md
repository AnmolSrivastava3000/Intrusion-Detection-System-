link to app-
🔐 IoT Intrusion Detection System (IDS) with Explainable AI
This repository contains an end-to-end Machine Learning pipeline and interactive web application for detecting anomalies and cyberattacks in IoT networks. The system is built using a highly optimized Random Forest classifier and features built-in Explainable AI (XAI) using SHAP to provide transparency into how predictions are made.

Developed by Anmol Srivastava as an 8th-semester B.Tech Computer Science and Engineering project.

🚀 Features
High-Accuracy Detection: Trained on the CICIoT2023 dataset, classifying network traffic as either 'Benign' or 'Attack' with exceptional accuracy and F1-scores.

Real-Time Web Interface: A lightweight, user-friendly frontend built with Streamlit that allows users to upload network traffic CSV files and get instant predictions.

Explainable AI (SHAP): Integrates shap.TreeExplainer to visualize feature importance, breaking down exactly which network features triggered an 'Attack' classification.

Dynamic Feature Alignment: Uses a configuration file (feature_config.json) to automatically validate and align incoming data features, ensuring pipeline stability.

🛠️ Tech Stack
Language: Python 3

Machine Learning: Scikit-Learn (Random Forest), XGBoost

Explainability: SHAP (SHapley Additive exPlanations)

Data Processing: Pandas, NumPy

Web Framework: Streamlit

Visualization: Matplotlib, Seaborn

📂 Repository Structure
├── app.py                      # Main Streamlit web application script
├── IDS_with_shap.ipynb         # Jupyter Notebook with data processing, model training, and SHAP logic
├── ids_pipeline.pkl            # Pre-trained Random Forest model
├── feature_config.json         # JSON file containing the exact features used during training
├── requirements.txt            # Python dependencies for deployment
├── sample_test.csv             # Sample dataset for users to test the live app
└── README.md                   # Project documentation

💻 Local Installation & Usage
To run this project on your local machine, follow these steps:

1. Clone the repository

Bash-

2. Install dependencies

Bash- pip install -r requirements.txt

3. Run the Streamlit App
4. 
Bash- streamlit run app.py

🌐 Cloud Deployment
This application is designed to be easily deployed on Streamlit Community Cloud. Simply link this GitHub repository to your Streamlit account, point it to app.py, and the platform will handle the rest using the provided requirements.txt file.

P.S.- i have added one of csv files from CICIoT dataset if you want to test it offline.
