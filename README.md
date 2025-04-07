# 🤰 Maternal Health Risk Predictor

This is a Streamlit-based web application that predicts maternal health risk (Low, Mid, or High) based on six clinical parameters. It uses a machine learning model (XGBoost) trained on a maternal health dataset, enhanced with data preprocessing, class balancing, and SHAP for model explainability.

---

## Features

- Interactive web UI for health data input
- Risk prediction using a trained XGBoost model
- SHAP plots showing feature contribution for each prediction
- Scaled input via StandardScaler to match training conditions

---

##  Input Features

- Age (Years)
- Systolic Blood Pressure (mmHg)
- Diastolic Blood Pressure (mmHg)
- Blood Sugar (mmol/L)
- Body Temperature (°F)
- Heart Rate (beats per minute)

---

##  How to Run

###  1. Clone the repository
```bash
git clone https://github.com/your-username/maternal-health-risk-app.git
cd maternal-health-risk-app
```

###  2. Install dependencies
```bash
pip install -r requirements.txt
```

###  3. Run the app
```bash
streamlit run app.py
```

---

## Files

| File                   | Description                             |
|------------------------|-----------------------------------------|
| `app.py`               | Main Streamlit web app                  |
| `xgboost_model.joblib` | Trained XGBoost model                   |
| `scaler.joblib`        | StandardScaler used during training     |
| `requirements.txt`     | Python dependencies                     |

---

## Model Details

- **Algorithm**: XGBoost Classifier
- **Preprocessing**: StandardScaler
- **Class Balancing**: SMOTE
- **Explainability**: SHAP waterfall plots

---

## Dataset Source

Maternal Health Risk Dataset  
[UCI Repository Link](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)

---

## License

This project is for academic and research use.
